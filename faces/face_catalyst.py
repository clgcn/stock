"""
催化剂/消息面 (CatalystFace)
============================
短线权重25% — 催化剂强度

分析内容:
  ① 个股新闻/行业新闻 — stock_news
  ② 公告/事件 — stock_announcements
  ③ 财报分析 — earnings_analysis (条件触发)

底层依赖:
  news_analyzer.get_stock_news_report()    → 个股新闻
  announcements.get_announcements()        → 公告列表
  announcements.format_announcements()     → 公告格式化
  data_fetcher.get_realtime()              → 获取股票名称
  financial.get_financial_history()        → 历史财务数据
  data_fetcher.get_kline()                 → K线 (财报反应)
  announcements.analyze_earnings_reaction()→ 财报市场反应
  announcements.format_earnings_analysis() → 财报分析报告
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from announcements import (
    get_announcements, format_announcements,
    analyze_earnings_reaction, format_earnings_analysis,
)
from data_fetcher import get_kline, get_kline_prefer_db, get_realtime
from financial import get_financial_history
import news_analyzer as na


# ══════════════════════════════════════════════════════
# 结构化信号
# ══════════════════════════════════════════════════════

@dataclass
class CatalystSignals:
    """催化剂/消息面结构化信号。"""

    # 催化剂强度 (短线权重25%)
    catalyst_type: str = "weak"              # "strong" / "medium" / "weak" / "negative"
    catalyst_detail: str = "无明显催化剂"

    # 个股新闻情绪
    stock_sentiment: float = 0.0             # -1.0 ~ +1.0
    news_highlights: List[str] = field(default_factory=list)

    # 公告事件
    has_earnings_report: bool = False        # 是否有财报类公告
    has_regulatory_issue: bool = False       # 是否有监管/问询函
    has_major_event: bool = False            # 是否有重大事件(回购/增持/减持)
    announcement_summary: str = ""

    # 财报质量 (条件触发)
    earnings_beat: bool = False
    earnings_miss: bool = False
    earnings_sentiment: float = 0.0          # -0.25 ~ +0.25


@dataclass
class CatalystResult:
    """催化剂面分析完整结果。"""
    signals: CatalystSignals
    stock_news_report: str = ""
    announcements_report: str = ""
    earnings_report: Optional[str] = None


# ══════════════════════════════════════════════════════
# 催化剂面类
# ══════════════════════════════════════════════════════

class CatalystFace:
    """
    催化剂/消息面分析模块。

    公开接口:
      analyze()     → CatalystResult   执行完整催化剂面分析
      score()       → list             催化剂强度(25%)
      check_veto()  → list             利空公告 / 监管问询

    内部方法:
      _fetch_news()                获取个股新闻
      _fetch_announcements()       获取公告列表
      _fetch_earnings_analysis()   执行财报分析 (条件触发)
      _extract_news()              从新闻报告提取情绪
      _extract_announcements()     从公告提取事件标记
      _extract_earnings()          从财报报告提取业绩信号
      _classify_catalyst_type()    综合判定催化剂强度
    """

    # ╔══════════════════════════════════════════════════╗
    # ║  公开接口                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def analyze(
        stock_code: str,
        stock_name: str = "",
        max_news: int = 8,
        local_only: bool = False,
    ) -> CatalystResult:
        """
        执行完整催化剂/消息面分析。

        Parameters
        ----------
        stock_code : str   股票代码
        stock_name : str   股票名称 (空则自动获取)
        max_news   : int   最大新闻条数
        local_only : bool  True=纯本地模式（选股用），K线只读DB
        """
        signals = CatalystSignals()
        news_report = ""
        ann_report = ""
        earn_report = None

        # ── 1. 个股新闻 ──
        news_report = CatalystFace._fetch_news(stock_code, stock_name, max_news)
        CatalystFace._extract_news(signals, news_report)

        # ── 2. 公告面 ──
        anns, ann_report, resolved_name = CatalystFace._fetch_announcements(stock_code, stock_name)
        if not stock_name:
            stock_name = resolved_name
        CatalystFace._extract_announcements(signals, anns, ann_report)

        # ── 3. 财报分析 (条件触发) ──
        if signals.has_earnings_report:
            earn_report = CatalystFace._fetch_earnings_analysis(stock_code, stock_name, local_only=local_only)
            CatalystFace._extract_earnings(signals, earn_report)

        # ── 综合催化剂强度判定 ──
        CatalystFace._classify_catalyst_type(signals)

        return CatalystResult(
            signals=signals,
            stock_news_report=news_report,
            announcements_report=ann_report,
            earnings_report=earn_report,
        )

    @staticmethod
    def score(signals: CatalystSignals) -> list:
        """
        催化剂面评分 → [催化剂强度(25%)]。

        评分规则:
          strong   (当日利好公告/业绩超预期)  → 9-10
          medium   (行业政策/预期改善)        → 5-8
          weak     (无明显催化剂)             → 2-4
          negative (利空公告/监管)            → 0-1
        """
        ct = signals.catalyst_type
        if ct == "strong":
            score = 9.5
        elif ct == "medium":
            score = 6.5
        elif ct == "negative":
            score = 0.5
        else:
            score = 3.0

        return [
            {"name": "催化剂强度", "score": round(score, 1), "weight": 0.25, "max": 10},
        ]

    @staticmethod
    def check_veto(signals: CatalystSignals) -> list:
        """
        催化剂面一票否决。

        触发条件:
          · 利空公告当日 → 催化剂强度≤1
          · 监管问询函   → 整体×0.8
        """
        vetos = []
        if signals.catalyst_type == "negative":
            vetos.append((True, "利空公告当日", 1.0))
        if signals.has_regulatory_issue:
            vetos.append((True, "近期收到监管问询函", None))
        return vetos

    # ╔══════════════════════════════════════════════════╗
    # ║  数据获取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _fetch_news(stock_code: str, stock_name: str, max_news: int) -> str:
        """获取个股新闻报告。"""
        try:
            name = stock_name or stock_code
            return na.get_stock_news_report(
                stock_name=name, stock_code=stock_code, max_items=max_news
            )
        except Exception as e:
            return f"个股新闻获取失败: {e}"

    @staticmethod
    def _fetch_announcements(stock_code: str, stock_name: str):
        """获取公告列表 + 格式化。Returns: (anns_list, report, resolved_name)"""
        try:
            anns = get_announcements(stock_code, page_size=20)
            # 获取股票名称
            resolved_name = stock_name
            if not resolved_name:
                try:
                    rt = get_realtime([stock_code])
                    resolved_name = rt.iloc[0]["name"] if not rt.empty else ""
                except Exception:
                    pass
            report = format_announcements(stock_code, anns, name=resolved_name)
            return anns, report, resolved_name
        except Exception as e:
            return [], f"公告获取失败: {e}", stock_name

    @staticmethod
    def _fetch_earnings_analysis(stock_code: str, stock_name: str, local_only: bool = False) -> str:
        """执行财报质量分析（含历史财务 + K线反应）。"""
        try:
            history = get_financial_history(stock_code, periods=8)
            if not history:
                return "未获取到历史财务数据"

            anns = get_announcements(stock_code, page_size=30)
            earnings_anns = [a for a in anns if a.get("is_earnings")][:6]

            kline_df = None
            if earnings_anns:
                from datetime import datetime, timedelta
                oldest = earnings_anns[-1].get("date", "")
                start_dt = (
                    (datetime.strptime(oldest, "%Y-%m-%d") - timedelta(days=40)).strftime("%Y-%m-%d")
                    if oldest else
                    (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")
                )
                try:
                    kline_df = get_kline_prefer_db(stock_code, period="daily", start=start_dt,
                                                    adjust="none", local_only=local_only)
                except Exception:
                    pass

            reactions = []
            for ann in earnings_anns:
                if ann.get("date"):
                    r = analyze_earnings_reaction(stock_code, ann["date"], kline_df)
                    r["title"] = ann.get("title", "")
                    reactions.append(r)

            return format_earnings_analysis(stock_code, stock_name, history, reactions)
        except Exception as e:
            return f"财报分析失败: {e}"

    # ╔══════════════════════════════════════════════════╗
    # ║  信号提取                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _extract_news(signals: CatalystSignals, report: str):
        """从个股新闻报告提取情绪值。"""
        m = re.search(r"个股新闻情绪微调[：:]\s*([+-]?\d+(?:\.\d+)?)", report)
        if m:
            signals.stock_sentiment = float(m.group(1))

    @staticmethod
    def _extract_announcements(signals: CatalystSignals, anns: list, report: str):
        """从公告列表提取: 财报/监管/重大事件标记 + 摘要。"""
        earnings_kw = ("年报", "半年报", "季报", "一季报", "三季报", "业绩预告", "业绩快报", "业绩修正")
        regulatory_kw = ("监管", "问询函", "警示函", "诉讼", "立案", "处罚")
        major_kw = ("回购", "增持", "减持", "重大", "收购", "合并")

        for ann in anns:
            title = ann.get("title", "")
            if any(kw in title for kw in earnings_kw):
                signals.has_earnings_report = True
            if any(kw in title for kw in regulatory_kw):
                signals.has_regulatory_issue = True
            if any(kw in title for kw in major_kw):
                signals.has_major_event = True

        if anns:
            signals.announcement_summary = "; ".join(
                a.get("title", "")[:30] for a in anns[:3]
            )

    @staticmethod
    def _extract_earnings(signals: CatalystSignals, report: str):
        """从财报分析报告提取: 业绩超预期/不及预期 + 情绪修正值。"""
        if not report:
            return
        if "news_sentiment 建议在技术分析基础上 +0.2 ~ +0.3" in report:
            signals.earnings_sentiment = 0.25
            signals.earnings_beat = True
        elif "news_sentiment 建议在技术分析基础上 -0.2 ~ -0.3" in report:
            signals.earnings_sentiment = -0.25
            signals.earnings_miss = True

    # ╔══════════════════════════════════════════════════╗
    # ║  综合判定                                        ║
    # ╚══════════════════════════════════════════════════╝

    @staticmethod
    def _classify_catalyst_type(signals: CatalystSignals):
        """
        综合判定催化剂强度: negative → strong → medium → weak。

        优先级: 负面 > 强利好 > 中等 > 弱/无
        """
        # 负面优先
        if signals.has_regulatory_issue:
            signals.catalyst_type = "negative"
            signals.catalyst_detail = "监管问询/处罚公告"
            return
        if signals.stock_sentiment <= -0.3 or signals.earnings_miss:
            signals.catalyst_type = "negative"
            signals.catalyst_detail = "利空消息/业绩不及预期"
            return

        # 强催化
        if signals.earnings_beat or signals.stock_sentiment >= 0.3:
            signals.catalyst_type = "strong"
            signals.catalyst_detail = "业绩超预期" if signals.earnings_beat else "强烈利好消息"
            return

        # 中等催化
        if signals.has_major_event or signals.stock_sentiment >= 0.1:
            signals.catalyst_type = "medium"
            signals.catalyst_detail = "重大事件/行业利好" if signals.has_major_event else "温和利好"
            return

        # 默认弱催化
        signals.catalyst_type = "weak"
        signals.catalyst_detail = "无明显催化剂"
