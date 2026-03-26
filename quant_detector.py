"""
量化资金活跃度检测模块 (quant_detector.py)
====================================================
通过三个维度的公开数据间接判断量化资金参与程度:

  维度一: 均单手数 (Avg Lot Size)
      量化资金为降低市场冲击会大量拆单 → 单笔成交手数极小
      数据来源: 东方财富逐笔成交 API（最近 N 笔）

  维度二: 分钟成交均匀度 (Volume Uniformity)
      量化VWAP/TWAP策略将成交均摊全天 → 分钟成交量变异系数低
      数据来源: 东方财富 1分钟K线

  维度三: 小单笔数占比 (Small Trade Ratio)
      高频量化倾向于 ≤5手的微单 → 微单占全部笔数的比例
      数据来源: 同逐笔数据

综合三个维度输出 0-100 的量化活跃度得分，并给出操作建议。

API 说明:
  - 逐笔接口: push2.eastmoney.com/api/qt/stock/details/get
  - 返回每笔: 时间, 价格, 手数, 金额, 方向(1=买入/2=卖出/4=中性)
  - 无需注册，免费公开
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from curl_cffi import requests
    _USE_CURL = True
except ImportError:
    import requests
    _USE_CURL = False

_NO_PROXY = {"http": "", "https": ""}


def _headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.eastmoney.com/",
    }


def _get(url, params=None, timeout=15):
    if _USE_CURL:
        return requests.get(url, params=params, headers=_headers(),
                            timeout=timeout, impersonate="chrome")
    return requests.get(url, params=params, headers=_headers(),
                        timeout=timeout, proxies=_NO_PROXY)


def _get_secid(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("60", "68", "51", "11")):
        return f"1.{code}"
    return f"0.{code}"


# ══════════════════════════════════════════════════════
# 一、逐笔成交数据获取
# ══════════════════════════════════════════════════════

def get_tick_data(code: str, n: int = 2000) -> pd.DataFrame:
    """
    从东方财富获取最近 n 笔逐笔成交数据。

    Parameters
    ----------
    code : str   股票代码，如 "600519"
    n    : int   获取最近 N 笔，上限约 2000（接口限制）

    Returns
    -------
    DataFrame: columns = [time, price, volume(手), amount(元), direction]
               direction: 1=主动买入  2=主动卖出  4=中性/集合竞价
    """
    secid = _get_secid(code)
    url = "https://push2.eastmoney.com/api/qt/stock/details/get"
    params = {
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55",
        "pos":     f"-{n}",
        "secid":   secid,
        "_":       int(time.time() * 1000),
    }
    try:
        r = _get(url, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ConnectionError(f"逐笔数据请求失败: {e}")

    raw = (data.get("data") or {}).get("details") or []
    if not raw:
        raise ValueError(f"未获取到 {code} 的逐笔数据（非交易时段或代码有误）")

    records = []
    for item in raw:
        parts = item.split(",")
        if len(parts) < 5:
            continue
        try:
            records.append({
                "time":      parts[0],
                "price":     float(parts[1]),
                "volume":    int(parts[2]),     # 单位: 手(100股)
                "amount":    float(parts[3]),   # 单位: 元
                "direction": int(parts[4]),     # 1=买 2=卖 4=中性
            })
        except (ValueError, IndexError):
            continue

    if not records:
        raise ValueError(f"逐笔数据解析失败，可能接口格式已变更")

    df = pd.DataFrame(records)
    return df


# ══════════════════════════════════════════════════════
# 二、分钟K线（用于成交均匀度分析）
# ══════════════════════════════════════════════════════

def get_minute_kline(code: str, period: str = "1m") -> pd.DataFrame:
    """
    获取当日或近期分钟K线，用于分析成交量时间分布。
    """
    # 复用 data_fetcher 的接口，避免重复代码
    from data_fetcher import get_kline
    df = get_kline(code, period=period, limit=250)
    return df


# ══════════════════════════════════════════════════════
# 三、核心指标计算
# ══════════════════════════════════════════════════════

def _calc_lot_metrics(tick_df: pd.DataFrame) -> dict:
    """
    基于逐笔数据计算手数相关指标。

    Returns dict:
      avg_lot     : 均单手数（越小量化可能性越高）
      median_lot  : 中位手数
      micro_ratio : ≤5手 的微单占比 (%)
      tiny_ratio  : ≤10手 的小单占比 (%)
      total_ticks : 总笔数
      buy_ratio   : 主动买入占比 (%)
    """
    vols = tick_df["volume"].values
    total = len(vols)

    avg_lot    = float(np.mean(vols))
    median_lot = float(np.median(vols))
    micro_ratio = float((vols <= 5).sum() / total * 100)
    tiny_ratio  = float((vols <= 10).sum() / total * 100)

    buy_count = (tick_df["direction"] == 1).sum()
    buy_ratio = float(buy_count / total * 100)

    return {
        "avg_lot":     round(avg_lot, 1),
        "median_lot":  round(median_lot, 1),
        "micro_ratio": round(micro_ratio, 1),
        "tiny_ratio":  round(tiny_ratio, 1),
        "total_ticks": total,
        "buy_ratio":   round(buy_ratio, 1),
    }


def _calc_volume_uniformity(minute_df: pd.DataFrame) -> dict:
    """
    基于分钟K线计算成交量均匀度。

    量化VWAP策略特征：
    - 全天成交量近似均匀（CV低）
    - 开盘/收盘放量程度较低（头尾比接近1）

    Returns dict:
      cv          : 成交量变异系数 (std/mean)，越低越均匀
      open_ratio  : 前30分钟成交量占比 (%)
      close_ratio : 后30分钟成交量占比 (%)
      mid_ratio   : 中间时段成交量占比 (%)
      uniformity  : 均匀度得分 0-100（越高越均匀）
    """
    vols = minute_df["volume"].values
    if len(vols) < 10:
        return {"cv": None, "open_ratio": None, "close_ratio": None,
                "mid_ratio": None, "uniformity": 50}

    total_vol = vols.sum()
    if total_vol == 0:
        return {"cv": None, "open_ratio": None, "close_ratio": None,
                "mid_ratio": None, "uniformity": 50}

    cv = float(np.std(vols) / np.mean(vols)) if np.mean(vols) > 0 else 1.0

    n = len(vols)
    head30 = min(30, n // 4)
    tail30 = min(30, n // 4)

    open_vol  = vols[:head30].sum()
    close_vol = vols[n - tail30:].sum()
    mid_vol   = vols[head30: n - tail30].sum()

    open_ratio  = float(open_vol / total_vol * 100)
    close_ratio = float(close_vol / total_vol * 100)
    mid_ratio   = float(mid_vol / total_vol * 100)

    # 均匀度评分：CV越低分数越高
    if cv < 0.3:   uniformity = 90
    elif cv < 0.5: uniformity = 75
    elif cv < 0.7: uniformity = 58
    elif cv < 1.0: uniformity = 40
    elif cv < 1.5: uniformity = 22
    else:          uniformity = 8

    return {
        "cv":           round(cv, 3),
        "open_ratio":   round(open_ratio, 1),
        "close_ratio":  round(close_ratio, 1),
        "mid_ratio":    round(mid_ratio, 1),
        "uniformity":   uniformity,
    }


# ══════════════════════════════════════════════════════
# 四、综合评分
# ══════════════════════════════════════════════════════

def _score_avg_lot(avg_lot: float) -> tuple[int, str]:
    """均单手数得分（满分40）"""
    if avg_lot < 3:
        return 40, "极小（<3手），高频拆单明显"
    elif avg_lot < 6:
        return 33, "很小（3-6手），疑似算法拆单"
    elif avg_lot < 12:
        return 24, "偏小（6-12手），有一定拆单迹象"
    elif avg_lot < 25:
        return 14, "适中（12-25手），量化参与有限"
    elif avg_lot < 60:
        return 6,  "偏大（25-60手），散户/游资为主"
    else:
        return 0,  "大单为主（>60手），机构主动建仓或大散户"


def _score_micro_ratio(ratio: float) -> tuple[int, str]:
    """微单（≤5手）占比得分（满分35）"""
    if ratio >= 70:
        return 35, f"{ratio:.0f}% 笔数为微单，高频量化特征强烈"
    elif ratio >= 55:
        return 28, f"{ratio:.0f}% 笔数为微单，量化参与明显"
    elif ratio >= 40:
        return 20, f"{ratio:.0f}% 笔数为微单，有量化参与"
    elif ratio >= 25:
        return 12, f"{ratio:.0f}% 笔数为微单，量化参与一般"
    else:
        return 4,  f"{ratio:.0f}% 笔数为微单，量化参与偏低"


def _score_uniformity(uniformity: int, cv: float) -> tuple[int, str]:
    """成交均匀度得分（满分25）"""
    score = int(uniformity * 25 / 100)
    if cv is None:
        return 12, "分钟数据不足，均匀度无法评估"
    if cv < 0.3:
        return 25, f"CV={cv:.2f}，成交量极其均匀，强烈VWAP/TWAP信号"
    elif cv < 0.5:
        return 20, f"CV={cv:.2f}，成交量较为均匀，算法策略特征"
    elif cv < 0.7:
        return 14, f"CV={cv:.2f}，成交量中等均匀，存在算法参与"
    elif cv < 1.0:
        return 8,  f"CV={cv:.2f}，成交量波动明显，情绪交易为主"
    else:
        return 3,  f"CV={cv:.2f}，成交量高度不均，散户/事件驱动为主"


def calc_quant_score(tick_metrics: dict, uniformity_metrics: dict) -> dict:
    """
    综合三个维度计算量化活跃度总分（0-100）。

    Returns
    -------
    dict:
      total_score    : 总分 0-100
      lot_score      : 均单手数得分（满分40）
      micro_score    : 微单占比得分（满分35）
      uniform_score  : 均匀度得分（满分25）
      level          : 等级文字
      lot_desc       : 均单描述
      micro_desc     : 微单描述
      uniform_desc   : 均匀度描述
    """
    lot_score,    lot_desc    = _score_avg_lot(tick_metrics["avg_lot"])
    micro_score,  micro_desc  = _score_micro_ratio(tick_metrics["micro_ratio"])
    uniform_score, uniform_desc = _score_uniformity(
        uniformity_metrics.get("uniformity", 50),
        uniformity_metrics.get("cv"),
    )

    total = lot_score + micro_score + uniform_score

    if total >= 80:
        level = "🔴 量化主导"
    elif total >= 62:
        level = "🟠 量化活跃"
    elif total >= 42:
        level = "🟡 量化参与"
    else:
        level = "🟢 量化偏少"

    return {
        "total_score":   total,
        "lot_score":     lot_score,
        "micro_score":   micro_score,
        "uniform_score": uniform_score,
        "level":         level,
        "lot_desc":      lot_desc,
        "micro_desc":    micro_desc,
        "uniform_desc":  uniform_desc,
    }


# ══════════════════════════════════════════════════════
# 五、操作建议
# ══════════════════════════════════════════════════════

def _build_advice(score: dict, tick: dict, uni: dict) -> list[str]:
    """
    根据量化活跃度给出针对性操作建议。
    """
    total = score["total_score"]
    lines = []

    if total >= 80:
        lines += [
            "· 量化资金高度主导，市场博弈以毫秒级机器对机器为主",
            "· 日内趋势容易被快速反转，不建议做短线频繁追涨杀跌",
            "· 适合跟随大趋势持仓，忽略分钟级噪音，以日线/4小时为操作周期",
            "· 挂单容易被扫，建议使用市价单或小幅让价成交",
            "· 止损位设在日线关键支撑，不要被分钟级波动轻易触发",
        ]
    elif total >= 62:
        lines += [
            "· 量化资金活跃，与散户/游资形成混合博弈",
            "· 盘中小波动较多，适合设置合理区间而非紧跟每次波动",
            "· 量化资金通常维护一定区间，突破区间上下沿时可考虑顺势操作",
            "· 建议不追涨停板（量化可能是打板后快速出货的一方）",
        ]
    elif total >= 42:
        lines += [
            "· 量化参与度中等，市场仍有较多情绪化交易机会",
            "· 消息面和技术面信号相对可信，可用常规策略操作",
            "· 注意开盘/收盘30分钟情绪博弈较激烈，非量化驱动",
        ]
    else:
        lines += [
            "· 量化参与偏低，市场以散户/游资情绪交易为主",
            "· 股价对消息面、题材更敏感，可关注催化剂驱动",
            "· 技术形态（趋势线、支撑阻力）较为有效",
            "· 流动性可能偏低，大资金入场需注意冲击成本",
        ]

    # 买卖力度提示
    buy_ratio = tick.get("buy_ratio", 50)
    if buy_ratio > 58:
        lines.append(f"· 当前主动买入占比 {buy_ratio:.0f}%，多方力量占优")
    elif buy_ratio < 42:
        lines.append(f"· 当前主动买入占比 {buy_ratio:.0f}%，空方压力偏大")

    # 收盘效应提示
    close_ratio = uni.get("close_ratio")
    if close_ratio and close_ratio > 25:
        lines.append(f"· 尾盘成交占比 {close_ratio:.0f}% 偏高，或有量化调仓/机构结算行为")

    return lines


# ══════════════════════════════════════════════════════
# 六、主报告入口
# ══════════════════════════════════════════════════════

def get_quant_activity_report(code: str, tick_n: int = 1500) -> str:
    """
    生成量化资金活跃度分析报告（中文）。

    Parameters
    ----------
    code   : str   A股代码，如 "600519"
    tick_n : int   分析最近 N 笔逐笔成交，默认 1500 笔

    Returns
    -------
    str  完整中文报告
    """
    from data_fetcher import get_realtime
    now = datetime.now()
    sep = "─" * 52

    # ── 获取股票名称 ──
    try:
        rt = get_realtime([code])
        name = rt.iloc[0]["name"] if not rt.empty else code
        current_price = rt.iloc[0]["current"]
        pct_chg = rt.iloc[0]["pct_chg"]
    except Exception:
        name = code
        current_price = None
        pct_chg = None

    lines = [
        "═" * 52,
        f"  量化资金活跃度分析  {name}（{code}）",
        f"  {now.strftime('%Y-%m-%d %H:%M')}",
        "═" * 52,
        "",
    ]

    if current_price:
        chg_str = f"  {'▲' if pct_chg >= 0 else '▼'} {pct_chg:+.2f}%"
        lines.append(f"  当前价格: {current_price:.2f}  {chg_str}")
        lines.append("")

    # ── 维度一&三: 逐笔数据 ──
    tick_metrics = None
    tick_error   = None
    try:
        tick_df = get_tick_data(code, n=tick_n)
        tick_metrics = _calc_lot_metrics(tick_df)
    except Exception as e:
        tick_error = str(e)

    # ── 维度二: 分钟均匀度 ──
    uni_metrics = {"cv": None, "open_ratio": None,
                   "close_ratio": None, "mid_ratio": None, "uniformity": 50}
    uni_error = None
    try:
        min_df = get_minute_kline(code, period="1m")
        if len(min_df) >= 10:
            uni_metrics = _calc_volume_uniformity(min_df)
    except Exception as e:
        uni_error = str(e)

    # ── 处理两种数据都失败的情况 ──
    if tick_metrics is None:
        lines.append(f"  ⚠️  逐笔数据获取失败（可能为非交易时段）")
        lines.append(f"  错误: {tick_error}")
        lines.append("")
        lines.append("  建议在交易时段（09:30-15:00）重新调用。")
        return "\n".join(lines)

    # ── 计算综合评分 ──
    score = calc_quant_score(tick_metrics, uni_metrics)

    # ── 输出: 综合评分 ──
    lines += [
        f"【综合量化活跃度得分】",
        "",
        f"  {score['total_score']:>3} / 100    {score['level']}",
        "",
        f"  {'█' * (score['total_score'] // 5)}{'░' * (20 - score['total_score'] // 5)}  {score['total_score']}%",
        "",
        sep,
    ]

    # ── 输出: 三维度明细 ──
    lines += [
        "【三维度详细分析】",
        "",
        f"  ① 均单手数  ({score['lot_score']:>2}/40分)",
        f"     均值: {tick_metrics['avg_lot']:.1f}手  |  中位: {tick_metrics['median_lot']:.1f}手",
        f"     {score['lot_desc']}",
        "",
        f"  ② 微单占比  ({score['micro_score']:>2}/35分)",
        f"     ≤5手微单: {tick_metrics['micro_ratio']:.1f}%  |  ≤10手小单: {tick_metrics['tiny_ratio']:.1f}%",
        f"     {score['micro_desc']}",
        "",
        f"  ③ 成交均匀度  ({score['uniform_score']:>2}/25分)",
    ]

    if uni_metrics["cv"] is not None:
        lines += [
            f"     变异系数CV: {uni_metrics['cv']:.3f}",
            f"     开盘30分: {uni_metrics['open_ratio']:.1f}%  |  中段: {uni_metrics['mid_ratio']:.1f}%  |  尾盘30分: {uni_metrics['close_ratio']:.1f}%",
            f"     {score['uniform_desc']}",
        ]
    else:
        lines.append(f"     {uni_error or '分钟数据不足，均匀度无法评估'}")

    # ── 输出: 买卖力道 ──
    lines += [
        "",
        sep,
        "【买卖力道】",
        "",
        f"  分析笔数: {tick_metrics['total_ticks']} 笔",
        f"  主动买入: {tick_metrics['buy_ratio']:.1f}%   "
        f"主动卖出: {100 - tick_metrics['buy_ratio']:.1f}%",
    ]

    buy = tick_metrics["buy_ratio"]
    if buy > 55:
        lines.append("  → 买方力量偏强，短期上行压力较小")
    elif buy < 45:
        lines.append("  → 卖方力量偏强，短期承压")
    else:
        lines.append("  → 买卖相对均衡")

    # ── 输出: 操作建议 ──
    advice = _build_advice(score, tick_metrics, uni_metrics)
    lines += [
        "",
        sep,
        "【针对性操作建议】",
        "",
    ]
    lines.extend(f"  {a}" for a in advice)

    lines += [
        "",
        "  ⚠️  本分析仅基于公开成交数据推断，不构成投资建议。",
        "═" * 52,
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 命令行调试
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    print(get_quant_activity_report(code))
