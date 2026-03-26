"""
财务数据模块 (financial) — 基本面依赖
======================================
提供:
  get_financial()          → 实时财务指标 (PE/PB/ROE等)
  get_financial_history()  → 历史多期财务数据
  get_balance_sheet()      → 资产负债表关键指标
  get_dividend_history()   → 历史分红派息记录
  compute_peg()            → PEG比率计算
"""

from _http_utils import _get, _get_secid
import time


# ──────────────────────────────────────────
# 1. 实时财务指标
# ──────────────────────────────────────────

def get_financial(code: str) -> dict:
    """Fetch key financial indicators from Eastmoney (PE, PB, ROE, etc.)."""
    secid = _get_secid(code)
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "invt": 2, "fltt": 2,
        "fields": ("f57,f58,f43,f44,f45,f46,f47,f48,f50,f57,f58,"
                    "f100,f116,f117,f162,f163,f164,f167,f168,f169,"
                    "f170,f171,f173,f174,f175,f177,f178"),
        "secid": secid, "_": int(time.time() * 1000),
    }
    try:
        resp = _get(url, params=params)
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"Financial data request failed: {e}")

    d = data.get("data", {}) or {}

    def safe(key, divisor=1):
        v = d.get(key, "-")
        if v in ["-", None, ""]:
            return None
        try:
            return round(float(v) / divisor, 4)
        except Exception:
            return v

    return {
        "code": code, "name": d.get("f58", code),
        "current": safe("f43"),
        "pe_ttm": safe("f162"), "pe_static": safe("f163"),
        "pb": safe("f167"), "ps_ttm": safe("f164"),
        "total_mv": safe("f116", 1e8), "float_mv": safe("f117", 1e8),
        "roe": safe("f173"), "gross_margin": safe("f174"),
        "net_margin": safe("f175"), "turnover_rate": safe("f168"),
        "volume_ratio": safe("f50"),
    }


# ──────────────────────────────────────────
# 2. 历史多期财务数据
# ──────────────────────────────────────────

def get_financial_history(code: str, periods: int = 8) -> list[dict]:
    """
    从东方财富获取个股历史多期财务数据（按报告期倒序）。

    Parameters
    ----------
    code    : str  股票代码，如 "000423"
    periods : int  获取期数，默认 8 期（约 2 年）

    Returns
    -------
    list of dict，每条包含:
      report_date : str   报告期 "YYYY-MM-DD"
      report_type : str   报告类型（年报/半年报/一季报/三季报）
      revenue     : float 营业收入（亿元）
      revenue_yoy : float 营收同比增长率 (%)
      net_profit  : float 归母净利润（亿元）
      profit_yoy  : float 净利润同比增长率 (%)
      eps         : float 基本每股收益（元）
      roe         : float 净资产收益率 (%)
      gross_margin: float 毛利率 (%)
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_LICO_FN_CPD",
        "columns":     "SECURITY_CODE,REPORTDATE,BASIC_EPS,"
                       "TOTAL_OPERATE_INCOME,YSTZ,PARENT_NETPROFIT,"
                       "SJLTZ,XSMLL,WEIGHTAVG_ROE",
        "filter":      f'(SECURITY_CODE="{code}")',
        "pageNumber":  1,
        "pageSize":    periods,
        "sortTypes":   -1,
        "sortColumns": "REPORTDATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"历史财务数据请求失败: {e}")

    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        raise ValueError(f"未获取到 {code} 的历史财务数据")

    def _report_type(date_str: str) -> str:
        if not date_str:
            return ""
        m = date_str[5:7]
        return {"03": "一季报", "06": "半年报", "09": "三季报", "12": "年报"}.get(m, "")

    def _to_yi(val) -> float | None:
        """转换为亿元，保留2位"""
        try:
            return round(float(val) / 1e8, 2) if val is not None else None
        except Exception:
            return None

    def _pct(val) -> float | None:
        try:
            return round(float(val), 2) if val is not None else None
        except Exception:
            return None

    results = []
    for row in rows:
        rd = (row.get("REPORTDATE") or "")[:10]
        results.append({
            "report_date":  rd,
            "report_type":  _report_type(rd),
            "eps":          _pct(row.get("BASIC_EPS")),
            "revenue":      _to_yi(row.get("TOTAL_OPERATE_INCOME")),
            "revenue_yoy":  _pct(row.get("YSTZ")),
            "net_profit":   _to_yi(row.get("PARENT_NETPROFIT")),
            "profit_yoy":   _pct(row.get("SJLTZ")),
            "gross_margin": _pct(row.get("XSMLL")),
            "roe":          _pct(row.get("WEIGHTAVG_ROE")),
        })
    return results


# ──────────────────────────────────────────
# 3. 资产负债表
# ──────────────────────────────────────────

def get_balance_sheet(code: str) -> dict:
    """
    获取最新资产负债表关键指标（资产负债率、商誉占净资产比）。

    Returns
    -------
    dict:
        report_date    : str   报告期
        total_assets   : float 总资产（亿元）
        total_liab     : float 总负债（亿元）
        equity         : float 净资产（归母，亿元）
        debt_ratio     : float 资产负债率 (%)
        goodwill       : float 商誉（亿元）
        goodwill_ratio : float 商誉/净资产 (%)
        cash           : float 货币资金（亿元）
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_DMSK_FN_BALANCE",
        "columns":     "SECURITY_CODE,REPORT_DATE,TOTAL_ASSETS,TOTAL_LIABILITIES,"
                       "PARENT_EQUITY,GOODWILL,MONETARYFUNDS",
        "filter":      f'(SECURITY_CODE="{code}")',
        "pageNumber":  1,
        "pageSize":    1,
        "sortTypes":   -1,
        "sortColumns": "REPORT_DATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"资产负债表请求失败: {e}")

    rows = (data.get("result") or {}).get("data") or []
    if not rows:
        raise ValueError(f"未获取到 {code} 的资产负债表数据")

    r = rows[0]
    def _yi(v):
        try: return round(float(v) / 1e8, 2) if v else None
        except: return None
    def _pct(a, b):
        try: return round(float(a) / float(b) * 100, 2) if a and b and float(b) != 0 else None
        except: return None

    total_assets = _yi(r.get("TOTAL_ASSETS"))
    total_liab   = _yi(r.get("TOTAL_LIABILITIES"))
    equity       = _yi(r.get("PARENT_EQUITY"))
    goodwill     = _yi(r.get("GOODWILL")) or 0.0

    return {
        "report_date":    (r.get("REPORT_DATE") or "")[:10],
        "total_assets":   total_assets,
        "total_liab":     total_liab,
        "equity":         equity,
        "debt_ratio":     _pct(total_liab, total_assets),
        "goodwill":       goodwill,
        "goodwill_ratio": _pct(goodwill * 1e8, (equity or 0) * 1e8) if equity else None,
        "cash":           _yi(r.get("MONETARYFUNDS")),
    }


# ──────────────────────────────────────────
# 4. 分红历史
# ──────────────────────────────────────────

def get_dividend_history(code: str, years: int = 5) -> list:
    """
    获取个股历史分红派息记录。

    Parameters
    ----------
    code  : str  股票代码
    years : int  获取最近 N 年的记录，默认 5 年

    Returns
    -------
    list of dict，每条包含:
        report_date   : str   报告期
        ann_date      : str   公告日
        ex_date       : str   除权除息日
        div_per_share : float 每股分红（元，税前）
        bonus_ratio   : float 送股比例（每10股送 X 股）
        total_div     : float 分红总额（亿元）
    """
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName":  "RPT_SHAREHOLDER_DIV_PLAN",
        "columns":     "SECURITY_CODE,PLAN_NOTICE_DATE,EX_DIVIDEND_DATE,"
                       "PER_CASH_DIV,BONUS_SHARE_RATIO,TOTAL_DIVIDEND,REPORT_DATE",
        "filter":      f'(SECURITY_CODE="{code}")',
        "pageNumber":  1,
        "pageSize":    years * 3,
        "sortTypes":   -1,
        "sortColumns": "PLAN_NOTICE_DATE",
        "source":      "WEB",
        "client":      "WEB",
    }
    try:
        resp = _get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"分红数据请求失败: {e}")

    rows = (data.get("result") or {}).get("data") or []

    def _f(v, d=4):
        try: return round(float(v), d) if v else None
        except: return None
    def _yi(v):
        try: return round(float(v) / 1e8, 2) if v else None
        except: return None

    results = []
    for r in rows:
        results.append({
            "report_date":   (r.get("REPORT_DATE") or "")[:10],
            "ann_date":      (r.get("PLAN_NOTICE_DATE") or "")[:10],
            "ex_date":       (r.get("EX_DIVIDEND_DATE") or "")[:10],
            "div_per_share": _f(r.get("PER_CASH_DIV"), 3),
            "bonus_ratio":   _f(r.get("BONUS_SHARE_RATIO"), 2),
            "total_div":     _yi(r.get("TOTAL_DIVIDEND")),
        })
    return results


# ──────────────────────────────────────────
# 5. PEG 计算
# ──────────────────────────────────────────

def compute_peg(financial_history: list, pe_ttm: float) -> dict:
    """
    计算 PEG 比率 (Price/Earnings to Growth)。

    PEG = PE_TTM / 净利润增速(%)
    PEG < 1  → 估值合理甚至低估
    PEG 1~2  → 合理区间
    PEG > 2  → 估值偏高
    PEG < 0  → 利润下滑，无参考意义

    Parameters
    ----------
    financial_history : list   来自 get_financial_history()
    pe_ttm            : float  当前PE(TTM)

    Returns
    -------
    dict: peg, growth_rate, growth_method, verdict
    """
    if not financial_history or pe_ttm is None or pe_ttm <= 0:
        return {"peg": None, "growth_rate": None, "growth_method": "数据不足", "verdict": "无法计算"}

    annual = [r for r in financial_history if r.get("report_type") == "年报"]
    recent = annual[:3] if annual else financial_history[:4]

    growths = [r.get("profit_yoy") for r in recent if r.get("profit_yoy") is not None]
    if not growths:
        return {"peg": None, "growth_rate": None, "growth_method": "无增速数据", "verdict": "无法计算"}

    avg_growth = sum(growths) / len(growths)
    method = f"近{len(growths)}期平均净利润增速"

    if avg_growth <= 0:
        verdict = "净利润下滑，PEG无参考意义（负增长）"
        return {"peg": None, "growth_rate": round(avg_growth, 2), "growth_method": method, "verdict": verdict}

    peg = round(pe_ttm / avg_growth, 3)

    if peg < 0.5:
        verdict = "PEG极低（<0.5），估值极度低估，成长性突出"
    elif peg < 1.0:
        verdict = "PEG<1，估值合理偏低，兼顾成长性的价值洼地"
    elif peg < 1.5:
        verdict = "PEG在1~1.5，估值合理"
    elif peg < 2.0:
        verdict = "PEG在1.5~2，估值偏高需关注"
    else:
        verdict = "PEG>2，估值较高，成长性不足以支撑当前PE"

    return {
        "peg":          peg,
        "growth_rate":  round(avg_growth, 2),
        "growth_method": method,
        "verdict":      verdict,
    }
