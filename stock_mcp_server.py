#!/usr/bin/env python3
"""
A-Share Stock Data MCP Server
================================
Enables Claude Desktop to directly call A-share real-time quotes,
K-line data, financial indicators, and technical analysis tools.

Installation:
    pip install mcp

Register in Claude Desktop config (see README or setup instructions).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# -- Add this directory to path for module imports --
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import db
from _http_utils import cn_now, next_trade_day, last_trade_day

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: mcp package not found. Please run: pip install mcp\n"
        "   or run: bash setup.sh",
        file=sys.stderr,
    )
    sys.exit(1)

import data_fetcher as df_mod
import financial as fin_mod
import capital_flow as cf_mod
import announcements as ann_mod
import market_quote as mq
import quant_engine as qe
import backtest_engine as bt
import risk_manager as rm
import stock_screener as sc
import news_analyzer as na
import quant_detector as qd
import slow_fetcher as sf

# ── 环境层 (架构图 Module A) ──
import module_a_environment as mod_a

# ── 五面体系 + 评分卡 (替代旧 B1/B2/C/D 模块) ──
from faces import (
    TechnicalFace,
    CapitalFace,
    CatalystFace,
    FundamentalFace,
    RiskFace,
    Scorecard,
    compute_combined_decision,
    format_combined_decision,
)

# -- Load instructions from prompt file --
_PROMPT_FILE = _HERE / "prompt.md"
if _PROMPT_FILE.exists():
    _instructions = _PROMPT_FILE.read_text(encoding="utf-8")
else:
    _instructions = "A股量化投资研究员 — 请确保 prompt.md 文件存在于 MCP Server 同级目录。"

# -- Initialize MCP server --
mcp = FastMCP(
    name="a-share-stock",
    instructions=_instructions,
)

# =====================================================
# Internal helpers
# =====================================================

def _resolve_stock_candidates(query: str, limit: int = 10):
    q = (query or "").strip()
    if not q:
        return []

    conn = sf._get_db()
    try:
        if q.isdigit() and len(q) == 6:
            return db.execute(conn,
                """
                SELECT code, name, suspended
                FROM stocks
                WHERE code = %s
                ORDER BY suspended ASC, code ASC
                LIMIT %s
                """,
                (q, max(int(limit), 1)),
            ).fetchall()

        exact_rows = db.execute(conn,
            """
            SELECT code, name, suspended
            FROM stocks
            WHERE name = %s
            ORDER BY suspended ASC, code ASC
            LIMIT %s
            """,
            (q, max(int(limit), 1)),
        ).fetchall()
        like_rows = db.execute(conn,
            """
            SELECT code, name, suspended
            FROM stocks
            WHERE name LIKE %s
            ORDER BY
                CASE WHEN name = %s THEN 0
                     WHEN name LIKE %s THEN 1
                     ELSE 2 END,
                suspended ASC,
                code ASC
            LIMIT %s
            """,
            (f"%{q}%", q, f"{q}%", max(int(limit), 1)),
        ).fetchall()

        seen = set()
        merged = []
        for row in list(exact_rows) + list(like_rows):
            key = row[0]
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        return merged
    finally:
        conn.close()


def _resolve_stock_unique(query: str):
    rows = _resolve_stock_candidates(query, limit=10)
    if len(rows) == 1:
        code, name, suspended = rows[0]
        return {"code": code, "name": name, "suspended": bool(suspended), "rows": rows}
    return {"code": None, "name": None, "suspended": None, "rows": rows}


# =====================================================
# Tool 1: Resolve Stock
# =====================================================

@mcp.tool()
def resolve_stock(query: str, limit: int = 10) -> str:
    """
    Resolve a stock name/code/alias-like input into canonical A-share code + name
    using the local stocks universe table.

    Parameters
    ----------
    query : str
        Stock code, exact name, or partial name. Example: "000001", "平安银行", "中国长城".
    limit : int
        Maximum number of matches to return, default 10.

    Returns
    -------
    A plain-text resolution result. When uniquely matched, returns the canonical
    stock code and name. When ambiguous, returns the top candidates and asks the
    caller to continue with the confirmed code.
    """
    q = (query or "").strip()
    if not q:
        return "ERROR: query is empty."

    try:
        rows = _resolve_stock_candidates(q, limit=limit)
    except Exception as e:
        return f"ERROR: Failed to resolve stock from local database: {e}"

    if not rows:
        return (
            f"No stock matched '{q}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )

    if len(rows) == 1:
        code, name, suspended = rows[0]
        suspended_text = "Yes" if suspended else "No"
        return (
            "Resolved Stock\n"
            "--------------\n"
            f"Query: {q}\n"
            f"Code: {code}\n"
            f"Name: {name}\n"
            f"Suspended: {suspended_text}\n"
            "Use this canonical code/name pair for all subsequent analysis."
        )

    lines = [
        "Ambiguous Stock Match",
        "----------------------",
        f"Query: {q}",
        f"Found {len(rows)} candidates. Use the exact code for the next step:",
    ]
    for code, name, suspended in rows[:max(int(limit), 1)]:
        tag = "停牌" if suspended else "正常"
        lines.append(f"- {code} {name} ({tag})")
    return "\n".join(lines)


# =====================================================
# Tool 2: Full Stock Analysis
# =====================================================

@mcp.tool()
def full_stock_analysis(
    stock_query: str,
    analysis_mode: str = "full",
    analysis_days: int = 250,
    monte_carlo_days: int = 20,
    include_quant_activity: bool = True,
    include_market_news: bool = True,
    include_northbound: bool = True,
    include_moneyflow: bool = True,
    include_valuation_quality: bool = True,
    include_balance_sheet: bool = True,
    include_margin_trading: bool = False,
    include_dividend_history: bool = False,
) -> str:
    """
    Run the complete single-stock analysis workflow in one call — all dimensions.

    Architecture: Module A → B1 (short) + B2 (long) → C (scorecard) → D (risk)

    This tool is the preferred entry point when the user asks:
    - "分析这只股票" / "这只股票能不能买" / "帮我看看某只票"

    Modular workflow (对应架构图):
      模块 A  环境感知层（每次必跑）: market_news + northbound_flow → 大盘评级 H/M/L
      模块 B1 短线分析: stock_news → moneyflow/margin → kline → diagnosis → quant_activity
      模块 B2 长线分析: financial_data → valuation_quality → balance_sheet → earnings → dividend
      模块 C  综合评分卡: 短线5维(0-100) + 长线6维(0-100) + 一票否决 + 亮点/风险
      模块 D  风控参数: risk_assessment → 止损/止盈/Kelly/VaR

    Parameters
    ----------
    stock_query          : str   股票代码或名称
    analysis_mode        : str   分析模式: "full" (短线+长线全跑) / "short" (仅短线) / "long" (仅长线)
    analysis_days        : int   诊断/风险分析的历史天数（默认250~1年）
    monte_carlo_days     : int   蒙特卡洛模拟未来天数（默认20交易日）
    include_quant_activity    : bool  是否调用量化活跃度（交易时段有效，默认True）
    include_market_news       : bool  是否调用市场新闻基线（默认True）
    include_northbound        : bool  是否调用北向资金（默认True）
    include_moneyflow         : bool  是否调用主力资金流向（默认True）
    include_valuation_quality : bool  是否调用PEG+杜邦分析（默认True）
    include_balance_sheet     : bool  是否调用负债率+商誉预警（默认True）
    include_margin_trading    : bool  是否调用融资融券余额（默认False，两融标的时开启）
    include_dividend_history  : bool  是否调用分红历史（默认False，价值型策略时开启）

    Returns
    -------
    完整的单股研究报告，包含环境评级、短线/长线分析、评分卡、风控参数。
    """
    resolved = _resolve_stock_unique(stock_query)
    rows = resolved["rows"]
    if not rows:
        return (
            f"ERROR: No stock matched '{stock_query}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )
    if len(rows) != 1:
        lines = [
            "ERROR: Stock query is ambiguous.",
            f"Query: {stock_query}",
            "Candidates:",
        ]
        for code, name, suspended in rows[:10]:
            tag = "停牌" if suspended else "正常"
            lines.append(f"- {code} {name} ({tag})")
        lines.append("Please retry with the exact code.")
        return "\n".join(lines)

    stock_code = resolved["code"]
    stock_name = resolved["name"]
    run_short = analysis_mode in ("full", "short")
    run_long = analysis_mode in ("full", "long")

    # ── ST/退市 前置拦截 ──────────────────────────────────────────────────
    is_st = any(tag in stock_name for tag in ("ST", "*ST", "△", "退"))
    if is_st:
        return (
            f"⚠️ ST/退市风险警示  {stock_name} ({stock_code})\n\n"
            "该标的处于 ST/退市风险区间（名称含 ST/*ST/退）:\n"
            "· 涨跌幅限制 ±5%，日内波动受限\n"
            "· 退市风险高，基本面因子模型不适用\n"
            "· 不建议建立中长线仓位\n\n"
            "如确认为壳资源重组行情，请在问题中注明'壳资源炒作'以获取专项分析。\n"
            "否则建议转换分析标的。"
        )
    # ────────────────────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════
    # 环境层 A — 市场评级 H/M/L（每次必跑）
    # ══════════════════════════════════════════════════
    env = mod_a.assess_environment(
        include_market_quote=include_market_news,
        include_news=include_market_news,
        include_northbound=include_northbound,
    )

    # ══════════════════════════════════════════════════
    # 各面分析 — 按 analysis_mode 选路径
    # ══════════════════════════════════════════════════

    # 催化剂面（短线+长线共用）
    catalyst_result = CatalystFace.analyze(
        stock_code=stock_code,
        stock_name=stock_name,
    )

    # 风控面（短线+长线共用）
    risk_result = RiskFace.analyze(
        stock_code=stock_code,
        analysis_days=analysis_days,
    )

    # ── 短线路径: 技术面 + 资金面 + 催化剂面 + 风控面 ──
    tech_result = None
    capital_result = None
    short_scorecard = None
    if run_short:
        tech_result = TechnicalFace.analyze(
            stock_code=stock_code,
            analysis_days=analysis_days,
            news_sentiment=env["combined_sentiment"],
            monte_carlo_days=monte_carlo_days,
        )
        capital_result = CapitalFace.analyze(
            stock_code=stock_code,
            include_margin=include_margin_trading,
            include_quant=include_quant_activity,
            northbound_adj=env["northbound_adj"],
        )
        short_scorecard = Scorecard.compute_short(
            tech_signals=tech_result.signals,
            capital_signals=capital_result.signals,
            catalyst_signals=catalyst_result.signals,
            risk_signals=risk_result.signals,
            market_rating=env["market_rating"],
        )

    # ── 长线路径: 基本面 + 催化剂面(降权) + 风控面 ──
    fundamental_result = None
    long_scorecard = None
    if run_long:
        fundamental_result = FundamentalFace.analyze(
            stock_code=stock_code,
            include_valuation=include_valuation_quality,
            include_balance=include_balance_sheet,
            include_dividend=include_dividend_history,
        )
        long_scorecard = Scorecard.compute_long(
            fundamental_signals=fundamental_result.signals,
            catalyst_signals=catalyst_result.signals,
            risk_signals=risk_result.signals,
            market_rating=env["market_rating"],
        )

    # ══════════════════════════════════════════════════
    # 组装完整报告
    # ══════════════════════════════════════════════════
    sections = [
        f"Full Stock Analysis  {stock_name} ({stock_code})",
        f"分析模式: {analysis_mode.upper()}  "
        f"报告时间: {cn_now().strftime('%Y-%m-%d %H:%M')}  "
        f"数据截止: {last_trade_day()}  "
        f"下一个交易日: {next_trade_day()}",
        "=" * 72,
        "",
        "[1] 股票解析",
        resolve_stock(stock_query, limit=10),
    ]

    # 环境层 A
    sections.extend(["", mod_a.format_environment_summary(env)])
    sections.extend(["", "[A-1] 市场环境详情", env["summary_text"]])
    news_rpt = env.get("news_report", "")
    if news_rpt:
        sections.extend(["", "[A-1.5] 市场新闻面", news_rpt])
    sections.extend(["", "[A-2] 北向资金详情", env["northbound_report"]])

    # 催化剂面（共用）
    sections.extend([
        "", "=" * 60,
        "【催化剂/消息面】",
        "=" * 60,
        catalyst_result.stock_news_report,
        "",
        catalyst_result.announcements_report,
    ])
    if catalyst_result.earnings_report:
        sections.extend(["", catalyst_result.earnings_report])

    # 短线路径
    if short_scorecard:
        sections.extend([
            "", "=" * 60,
            "【短线分析】技术面 + 资金面",
            "=" * 60,
        ])
        # 技术面
        sections.extend([
            "", "▌ 技术面",
            tech_result.kline_report,
            "", tech_result.diagnosis_report,
        ])
        # 资金面
        sections.extend([
            "", "▌ 资金面",
            capital_result.moneyflow_report,
        ])
        if capital_result.margin_report:
            sections.extend(["", capital_result.margin_report])
        if capital_result.quant_report:
            sections.extend(["", capital_result.quant_report])
        # 短线评分卡
        sections.extend(["", Scorecard.format_report(short_scorecard)])

    # 长线路径
    if long_scorecard:
        sections.extend([
            "", "=" * 60,
            "【长线分析】基本面",
            "=" * 60,
        ])
        sections.extend([
            "", "▌ 财务与估值",
            fundamental_result.financial_report,
            "", fundamental_result.valuation_report,
        ])
        if fundamental_result.balance_report:
            sections.extend(["", "▌ 资产负债", fundamental_result.balance_report])
        if fundamental_result.dividend_report:
            sections.extend(["", "▌ 分红历史", fundamental_result.dividend_report])
        # 长线评分卡
        sections.extend(["", Scorecard.format_report(long_scorecard)])

    # 风控面
    sections.extend([
        "", "=" * 60,
        "【风控面】",
        "=" * 60,
        risk_result.risk_report,
    ])

    # ── 建仓识别（四维度）──
    accum_result = None
    try:
        kline_df = sf.load_stock_history(stock_code)
        if not kline_df.empty and len(kline_df) >= 30:
            mf_data = sf.load_stock_moneyflow(stock_code, days=60)
            mg_trend = None
            if capital_result:
                mg_trend = getattr(capital_result.signals, "margin_trend", None)
            # 获取流通市值用于资金阈值相对化
            overview = sf.load_stock_overview(stock_code)
            float_mv = overview.get("float_mv", 0) or 0
            accum_result = qe.detect_accumulation(
                df=kline_df,
                moneyflow=mf_data if mf_data else None,
                margin_trend=mg_trend,
                northbound_adj=env["northbound_adj"],
                float_mv=float_mv,
            )
            if accum_result:
                sections.extend([
                    "", "=" * 60,
                    "【建仓识别】四维度分析",
                    "=" * 60,
                    accum_result["summary"],
                ])
    except Exception:
        pass

    # ── 综合决策（短线×长线 3×3 矩阵）──
    if short_scorecard and long_scorecard:
        try:
            # 尝试从技术面获取最新收盘价
            rt_price = 0.0
            try:
                rt = df_mod.get_realtime([stock_code])
                if rt is not None and not rt.empty:
                    rt_price = float(rt.iloc[0].get("current", 0) or 0)
            except Exception:
                pass
            combined = compute_combined_decision(
                short_scorecard, long_scorecard,
                risk_signals=risk_result.signals,
                current_price=rt_price,
            )
            sections.extend([
                "", "=" * 60,
                "【综合决策】短线进场 × 长线安全垫",
                "=" * 60,
                format_combined_decision(combined),
            ])
        except Exception:
            pass  # 不影响主报告
    elif short_scorecard:
        sections.extend([
            "", "─" * 50,
            "  ⚠️ 仅短线分析，无长线安全垫评级",
            "  建议按 C 档（无托底）规则管理仓位和止损",
            "─" * 50,
        ])

    return "\n".join(sections)


# =====================================================
# Tool 3: Full Stock Selection
# =====================================================

@mcp.tool()
def full_stock_selection(
    strategy: str = "auto",
    top_n: int = 10,
    review_top_n: int = 0,
    quality_threshold: float = 5.0,
    analysis_days: int = 250,
    monte_carlo_days: int = 20,
    include_quant_activity: bool = False,
    include_market_news: bool = True,
    include_margin_trading: bool = False,
    use_v2_pipeline: bool = False,
    v2_quality_quantile: float = 0.20,
    v2_min_keep: int = 3,
) -> str:
    """
    ★★★ 选股唯一入口 ★★★
    用户要求选股、找股票、扫描市场、给推荐名单时，必须调用本工具。
    不要用 stock_screener 代替——stock_screener 只是底层粗筛，不含北向资金/新闻面/财务分析/风控。

    A股全流程选股一体化工具（两阶段架构：本地DB预筛 → 实时深度复核）。
    内置: 环境感知(market_news+northbound_flow) → 本地DB粗筛 → 五面深度分析 → 评分卡 → 风控

    ══════════════════════════════════════════════════
    两阶段架构:
    ──────────────────────────────────────────────────
    第一阶段（本地DB筛选，0次网络API）:
      → 从本地 SQLite 读取全量 ~5000 只股票快照
      → 历史横截面评分（stock_history：趋势/回撤/量能）
      → 基础过滤（PE/PB/市值/换手率，按策略）
      → 7维量化深度扫描（本地kline，0次网络）
      → 质量阈值过滤（entry_quality_score ≥ quality_threshold）
      → 产出：质量合格候选股列表（通常10~25只）

    第二阶段（实时深度复核，每只股票全维度分析，与full_stock_analysis一致）:
      → 实时报价（realtime_quote）
      → 新闻面（stock_news）
      → 公告面（stock_announcements）
      → 财报分析（earnings_analysis，条件触发）
      → 财务数据（financial_data）
      → PEG + 杜邦分析（valuation_quality）
      → 资产负债 + 商誉预警（balance_sheet）
      → 分红历史（dividend_history，价值策略）
      → 主力资金流向（moneyflow）
      → 融资融券（margin_trading，可选）
      → K线技术指标（kline_data）
      → 量化诊断（stock_diagnosis，拉取最新实时K线）
      → 量化活跃度（quant_activity，可选）
      → 风险评估（risk_assessment）
    ══════════════════════════════════════════════════

    Parameters
    ----------
    strategy : str
        auto (默认，推荐) — 自动运行 value+momentum+oversold+potential 四策略，合并去重，综合排序
        value / momentum / oversold / potential / custom — 单策略模式
    top_n : int
        第一阶段筛选器内部返回候选数（建议 ≤20，影响候选池宽度）
    review_top_n : int
        第二阶段深度复核只数，默认0=分析所有通过质量门槛的候选股（并行执行，安全上限15只）
        ⚠️ 建议保持默认0，让系统自动决定。设为正数会限制深度复核数量，仪表盘也只展示复核过的股票。
        仅在用户明确要求"只看前N只"时才设正数
    quality_threshold : float
        【v1 only】第一阶段质量门槛（entry_quality_score，默认5.0）
        牛市可调高到10~15，熊市/震荡市可调低到0~3
    use_v2_pipeline : bool
        【推荐 True】使用 v2 量化规范管线(P0 修复):
          - 因子先横截面 z-score 再加权(EQS_V2_WEIGHTS)
          - 质量门槛改为横截面分位数(默认 top 20%)而非绝对值
          - momentum 策略允许涨停候选
          - auto 模式按"命中次数 + z-score 均值"合并,而非简单取最大
          - 精选阶段去掉 65%/55%/兜底瀑布,改 z-score 复合排名
    v2_quality_quantile : float
        【v2 only】Stage 2 质量分位门槛,默认 0.20(保留 top 20%)
        熊市可调至 0.30,强势市场可调至 0.10
    v2_min_keep : int
        【v2 only】分位截取最少保留数,默认 3(避免某策略结果极少时空盘)
    analysis_days : int
        诊断用历史天数
    monte_carlo_days : int
        蒙特卡洛模拟天数
    include_quant_activity : bool
        是否调用量化活跃度（仅交易时段有意义，默认False）
    include_market_news : bool
        是否调用市场新闻基线

    Returns
    -------
    四段式选股报告: 市场环境 + 候选池摘要 + 仪表盘 + 深度复核

    ⚠️ 展示要求（必须遵守）:
    本工具返回的所有深度复核股票都必须完整展示给用户，不允许只挑几只做精选摘要。
    每只候选股都应列出: 股票名称/代码、当前价格与涨跌幅、短线/长线评分、综合决策、
    建议仓位、蒙特卡洛上涨概率、核心风险点、操作建议。
    用户花时间等待全量分析，就是为了看到完整结果，而不是被二次筛选后只看3只。
    """
    top_n        = min(int(top_n), 30)
    review_top_n = int(review_top_n)
    if review_top_n > 0:
        review_top_n = min(review_top_n, 20)   # 手动指定时上限20

    # ════ 模块 A: 环境感知层 ════
    env = mod_a.assess_environment(
        include_market_quote=include_market_news,
        include_news=include_market_news,
        include_northbound=True,
    )
    market_report = env["summary_text"]
    northbound_report = env["northbound_report"]
    northbound_adj = env["northbound_adj"]
    news_report = env.get("news_report", "")

    # ════ 第一阶段: 纯本地DB筛选，0次外部API ════
    # screen_stocks 内部: SQLite读取 → 历史评分 → 快速过滤 → 本地kline量化扫描 → 质量阈值

    AUTO_STRATEGIES = ["value", "momentum", "oversold", "potential"]
    run_strategies = AUTO_STRATEGIES if strategy == "auto" else [strategy]
    strategy_label = "AUTO (value+momentum+oversold+potential)" if strategy == "auto" else strategy.upper()

    all_picks = {}  # code → best result dict (v1: 去重取最高 EQS)
    per_strategy_results = {}  # v2: 按策略保留全部,用于 hit-count 合并
    total_market = 0
    total_stage1 = 0
    total_stage2 = 0
    total_hist = 0
    strategy_stats = []  # 每个策略的产出统计
    data_health = None  # 由 screen_stocks/v2 返回的数据健康诊断 (全局同源)

    for strat in run_strategies:
        if use_v2_pipeline:
            sr = sc.screen_stocks_v2(
                strategy=strat,
                top_n=top_n,
                stage1_limit=min(max(top_n * 4, 60), 80),
                quality_quantile=v2_quality_quantile,
                min_keep=v2_min_keep,
                max_candidates=max(review_top_n + 5, 25) if review_top_n > 0 else 25,
            )
        else:
            sr = sc.screen_stocks(
                strategy=strat,
                top_n=top_n,
                deep_scan_enabled=True,
                stage1_limit=min(max(top_n * 4, 60), 80),
                quality_threshold=quality_threshold,
                max_candidates=max(review_top_n + 5, 25) if review_top_n > 0 else 25,
            )
        if "error" in sr:
            strategy_stats.append(f"  {strat}: 失败 — {sr['error']}")
            continue
        results = sr.get("results", [])
        total_market = max(total_market, sr.get("total_market", 0))
        total_stage1 += sr.get("stage1_count", 0)
        total_stage2 += sr.get("stage2_count", 0)
        total_hist = max(total_hist, sr.get("history_profile_count", 0))
        # 数据健康诊断: 所有策略跑同一份全市场快照, 取一次就够
        if data_health is None and sr.get("data_health"):
            data_health = sr.get("data_health")
        strategy_stats.append(f"  {strat}: {len(results)} 只通过质量门槛")

        # v2 path: keep per-strategy results intact for hit-count merge
        per_strategy_results[strat] = [dict(r) for r in results]

        # v1 path: dedupe by max EQS
        for r in results:
            code = r["code"]
            r["source_strategy"] = strat
            if code not in all_picks or (r.get("entry_quality_score", 0) or 0) > (all_picks[code].get("entry_quality_score", 0) or 0):
                all_picks[code] = r

    if use_v2_pipeline:
        # v2 merge: hit_count primary, avg z-score secondary
        picks = sc._merge_auto_results_v2(per_strategy_results)
    else:
        picks = sorted(all_picks.values(),
                       key=lambda x: x.get("entry_quality_score", 0) or 0,
                       reverse=True)
    qualified_count = len(picks)

    if not picks and strategy != "auto":
        # 单策略失败时直接返回错误
        screen_result = sc.screen_stocks(strategy=strategy, top_n=top_n, deep_scan_enabled=True)
        if "error" in screen_result:
            return f"ERROR: {screen_result['error']}"

    # ════ 批量拉取实时行情，覆盖本地DB旧价格 ════
    if picks:
        try:
            rt_codes = [p["code"] for p in picks]
            rt_df = df_mod.get_realtime(rt_codes)
            if rt_df is not None and not rt_df.empty:
                rt_map = {}
                for _, row in rt_df.iterrows():
                    c = str(row.get("code", ""))
                    cur = float(row.get("current", 0) or 0)
                    if cur > 0:
                        rt_map[c] = {
                            "current": cur,
                            "pct_chg": float(row.get("pct_chg", 0) or 0),
                            "open":    float(row.get("open", 0) or 0),
                            "high":    float(row.get("high", 0) or 0),
                            "low":     float(row.get("low", 0) or 0),
                            "volume":  float(row.get("volume", 0) or 0),
                            "amount":  float(row.get("amount", 0) or 0),
                        }
                for p in picks:
                    rt = rt_map.get(p["code"])
                    if rt:
                        p.update(rt)
        except Exception:
            pass  # 实时行情批量拉取失败，继续用本地DB数据

    # 生成筛选摘要文本（在实时行情更新后生成，确保价格是最新的）
    screen_report = sc._format_summary(
        strategy_label,
        total_market,
        total_stage1,
        total_stage2,
        picks,
        True,
        history_profile_count=total_hist,
        quality_threshold=quality_threshold,
    )
    # auto 模式追加各策略统计
    if strategy == "auto" and strategy_stats:
        screen_report += "\n\n各策略产出:\n" + "\n".join(strategy_stats)

    if not picks:
        empty_lines = [
            f"Full Stock Selection — {strategy_label}",
            "=" * 72,
            "",
        ]
        if data_health and data_health.get("is_stale"):
            empty_lines.extend([
                "⚠️  数据健康告警",
                f"  pct_chg覆盖率: {data_health.get('pct_chg_coverage', 0)*100:.1f}%   "
                f"最近交易日: {data_health.get('last_trade_date')}   "
                f"距今: {data_health.get('staleness_days', '?')} 天",
            ])
            for w in data_health.get("warnings", []):
                empty_lines.append(f"  • {w}")
            empty_lines.append("")
        empty_lines.extend([
            "【1】市场环境 / 宏观基线",
            market_report,
            "",
            "【2】第一阶段筛选结果（本地DB，0次网络请求）",
            screen_report,
            "",
            f"第一阶段未筛出符合质量阈值(≥{quality_threshold:.1f})的候选股。",
            "建议: 降低 quality_threshold（如3.0），或更换策略（如 oversold / potential）",
        ])
        return "\n".join(empty_lines)

    # ════ 精选门槛：从候选池中筛出"最可能上涨"的股票 ════
    MAX_REVIEW = 15
    pre_filter_count = len(picks)

    if use_v2_pipeline:
        # v2: single quantile cut over composite z-score of
        # (prob_up, total_score, expected_return). No cascades, no
        # magic thresholds. Top MAX_REVIEW by relative rank.
        refine_out = sc._refine_picks_v2(picks, max_review=MAX_REVIEW, min_review=5)
        picks_for_review = refine_out["picks_for_review"]
    elif len(picks) > MAX_REVIEW:
        # v1: original 65%/55%/fallback cascade
        refined = [p for p in picks if
            (p.get("prob_up") or 0) >= 65
            and (p.get("total_score") or 0) > 0
            and (p.get("expected_return") or 0) > 0
        ]
        if len(refined) < 5:
            refined = [p for p in picks if (p.get("prob_up") or 0) >= 55]
        if len(refined) > MAX_REVIEW:
            refined = refined[:MAX_REVIEW]
        if not refined:
            refined = picks[:MAX_REVIEW]
        picks_for_review = refined
    else:
        picks_for_review = picks

    actual_review = len(picks_for_review) if review_top_n <= 0 else max(1, min(review_top_n, len(picks_for_review)))
    actual_review = min(actual_review, MAX_REVIEW)

    # 精选统计（追加到 screen_report）
    if pre_filter_count > len(picks_for_review):
        if use_v2_pipeline:
            screen_report += (
                f"\n\n精选门槛(v2): 横截面 z-score((prob_up, total_score, expected_return)) 取前 {MAX_REVIEW}"
                f"\n  候选池: {pre_filter_count} 只 → 精选后: {len(picks_for_review)} 只进入深度复核"
            )
        else:
            screen_report += (
                f"\n\n精选门槛(v1): prob_up≥65% + total_score>0 + expected_return>0"
                f"\n  EQS候选: {pre_filter_count} 只 → 精选后: {len(picks_for_review)} 只进入深度复核"
            )

    # ════ 第二阶段: 深度复核（所有候选股并行分析）════

    # ── 辅助函数: 本地DB历史K线 + 实时补当天 ──
    def _local_kline(code: str, days: int = 10, rt_row: dict = None) -> str:
        """
        从本地 stock_history 读取历史K线，如果DB缺当天数据则用 rt_row 补上。
        rt_row: realtime quote dict {open, high, low, current, pct_chg, volume, amount}
                传 None 时仅用本地数据（不发额外API）。
        """
        try:
            import pandas as pd
            df = sf.load_stock_history(code)
            if df.empty:
                return f"{code} 本地DB无K线数据"
            df = df.sort_values("date")

            today_str = cn_now().strftime("%Y-%m-%d")
            db_last = df.iloc[-1]["date"]
            db_last_str = db_last.strftime("%Y-%m-%d") if hasattr(db_last, "strftime") else str(db_last)[:10]
            source_tag = "local_db"

            if db_last_str < today_str and rt_row is not None:
                try:
                    op = float(rt_row.get("open", 0))
                    if op > 0:
                        today_bar = pd.DataFrame([{
                            "date": pd.Timestamp(today_str),
                            "open": op,
                            "high": float(rt_row["high"]),
                            "low":  float(rt_row["low"]),
                            "close": float(rt_row["current"]),
                            "pct_chg": float(rt_row.get("pct_chg", 0)),
                            "volume": float(rt_row.get("volume", 0)),
                            "amount": float(rt_row.get("amount", 0)),
                        }])
                        df = pd.concat([df, today_bar], ignore_index=True)
                        source_tag = "local_db+realtime"
                except Exception:
                    pass

            df = df.tail(days).copy()
            _fmt = lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            lines_k = [
                f"{code} ({code}) daily K-line   Total {len(df)} bars  [{source_tag}]",
                f"   Range: {_fmt(df.iloc[0]['date'])} -> {_fmt(df.iloc[-1]['date'])}",
                "-" * 70,
            ]
            show_cols = ["date", "open", "high", "low", "close", "pct_chg", "volume"]
            disp = df[show_cols].copy()
            disp["date"] = disp["date"].apply(_fmt)
            lines_k.append(disp.to_string(index=False))
            latest = df.iloc[-1]
            lines_k.append("-" * 70)
            lines_k.append(f"Latest Close: {latest['close']:.2f}  Change: {latest['pct_chg']:+.2f}%")
            return "\n".join(lines_k)
        except Exception as e:
            return f"本地K线读取失败: {e}"

    def _safe(fn, *a, label="", **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            return f"{label} 获取失败: {e}"

    # ── 单只候选股全维度分析（面模块，在线程中执行）──
    # K 线数据全部走本地 DB（local_only=True），盘前选股零 API 调用。
    # 单股深度分析（full_stock_analysis）才走实时 API。
    def _analyze_one(idx: int, item: dict) -> dict:
        code = item["code"]
        name = item["name"]

        # 注: picks 已在第一阶段后批量更新过实时行情，item 中的 current/pct_chg 已是最新值

        # ═ 催化剂面 ═
        try:
            cat = CatalystFace.analyze(stock_code=code, stock_name=name, local_only=True)
        except Exception as _e:
            from faces.face_catalyst import CatalystResult, CatalystSignals
            cat = CatalystResult(signals=CatalystSignals(), stock_news_report=f"催化剂分析失败: {_e}",
                                 announcements_report="", earnings_report="")

        # ═ 技术面 (短线) ═
        try:
            tech = TechnicalFace.analyze(
                stock_code=code,
                analysis_days=analysis_days,
                news_sentiment=env["combined_sentiment"],
                monte_carlo_days=monte_carlo_days,
                local_only=True,
            )
        except Exception as _e:
            from faces.face_technical import TechnicalResult, TechnicalSignals
            tech = TechnicalResult(signals=TechnicalSignals(), kline_report=f"技术面分析失败: {_e}",
                                   diagnosis_report="")

        # ═ 资金面 (短线) ═
        try:
            cap = CapitalFace.analyze(
                stock_code=code,
                include_margin=include_margin_trading,
                include_quant=include_quant_activity,
                northbound_adj=env["northbound_adj"],
                local_only=True,
            )
        except Exception as _e:
            from faces.face_capital import CapitalResult, CapitalSignals
            cap = CapitalResult(signals=CapitalSignals(), moneyflow_report=f"资金面分析失败: {_e}",
                                margin_report="", quant_report="")

        # ═ 基本面 (长线) ═
        try:
            funda = FundamentalFace.analyze(
                stock_code=code,
                include_valuation=True,
                include_balance=True,
                include_dividend=strategy in ("value",),
                local_only=True,
            )
        except Exception as _e:
            from faces.face_fundamental import FundamentalResult, FundamentalSignals
            funda = FundamentalResult(signals=FundamentalSignals(), financial_report=f"基本面分析失败: {_e}",
                                      valuation_report="", balance_report="", dividend_report="")

        # ═ 风控面 ═
        try:
            risk = RiskFace.analyze(stock_code=code, analysis_days=analysis_days, local_only=True)
        except Exception as _e:
            from faces.face_risk import RiskResult, RiskSignals
            risk = RiskResult(signals=RiskSignals(), risk_report=f"风控分析失败: {_e}")

        # ═ 建仓识别 (四维) ═
        accum_result = None
        try:
            kline_df = sf.load_stock_history(code)
            if not kline_df.empty and len(kline_df) >= 30:
                mf_data = sf.load_stock_moneyflow(code, days=60)
                mg_trend = getattr(cap.signals, "margin_trend", None)
                # 传入流通市值用于资金流向相对化
                float_mv = item.get("float_mv", 0) or 0
                accum_result = qe.detect_accumulation(
                    df=kline_df,
                    moneyflow=mf_data if mf_data else None,
                    margin_trend=mg_trend,
                    northbound_adj=env["northbound_adj"],
                    float_mv=float_mv,
                )
        except Exception:
            pass

        # ═ 评分卡 ═
        sc_short = Scorecard.compute_short(
            tech_signals=tech.signals,
            capital_signals=cap.signals,
            catalyst_signals=cat.signals,
            risk_signals=risk.signals,
            market_rating=env["market_rating"],
        )
        sc_long = Scorecard.compute_long(
            fundamental_signals=funda.signals,
            catalyst_signals=cat.signals,
            risk_signals=risk.signals,
            market_rating=env["market_rating"],
        )

        # ═ 综合决策（短线×长线 3×3 矩阵）═
        rt_price = item.get("current", 0)
        rt_price_f = float(rt_price) if isinstance(rt_price, (int, float)) else 0.0
        combined = compute_combined_decision(
            sc_short, sc_long,
            risk_signals=risk.signals,
            current_price=rt_price_f,
        )

        # 构建仪表盘行
        rt_chg = item.get("pct_chg", "N/A")
        short_s = sc_short.final_total
        long_s = sc_long.final_total
        tier_tag = combined.long_tier
        action_tag = combined.matrix_action
        # 建仓信号标记
        accum_tag = ""
        if accum_result:
            ac = accum_result["conclusion"]
            if ac == "HIGH":
                accum_tag = " 🔴建仓"
            elif ac == "WATCH":
                accum_tag = " 🟡观察"
        if isinstance(rt_chg, float):
            dash = (f"| {idx} | {name} ({code}) "
                    f"| 短{short_s:.0f}/长{long_s:.0f}({tier_tag}档) "
                    f"| {action_tag}{accum_tag} "
                    f"| {combined.position_pct:.0f}% "
                    f"| {item.get('prob_up', 'N/A')} "
                    f"| {rt_price} ({rt_chg:+.2f}%) |")
        else:
            dash = (f"| {idx} | {name} ({code}) "
                    f"| 短{short_s:.0f}/长{long_s:.0f}({tier_tag}档) "
                    f"| {action_tag}{accum_tag} "
                    f"| {combined.position_pct:.0f}% "
                    f"| {item.get('prob_up', 'N/A')} "
                    f"| {rt_price} |")

        # 构建报告段落
        sec = [
            "",
            f"【候选 {idx}】{name}（{code}）",
            "-" * 72,
            f"  实时价格: {rt_price}"
            + (f" ({rt_chg:+.2f}%)" if isinstance(rt_chg, float) else ""),
            f"  历史得分: {item.get('history_score', 0):+.1f}   "
            f"Stage1: {item.get('stage1_score', 0):+.1f}   "
            f"综合质量分(EQS): {item.get('entry_quality_score', 0):+.1f}",
        ]
        # 综合决策（最重要，放最前面）
        sec.append(format_combined_decision(combined))
        # 短线评分卡
        sec.append(Scorecard.format_report(sc_short))
        # 长线评分卡
        sec.append(Scorecard.format_report(sc_long))
        # 风控详情
        sec.append(risk.risk_report)
        # 建仓识别
        if accum_result:
            sec.append(accum_result["summary"])

        return {"idx": idx, "dashboard": dash, "sections": sec,
                "combined": combined, "accum": accum_result}

    # ── 所有候选股并行执行 ──
    review_results = []
    with ThreadPoolExecutor(max_workers=min(actual_review, 8)) as outer_pool:
        futures = {
            outer_pool.submit(_analyze_one, idx, item): idx
            for idx, item in enumerate(picks_for_review[:actual_review], 1)
        }
        for fut in as_completed(futures, timeout=300):
            try:
                review_results.append(fut.result(timeout=25))
            except Exception as e:
                i = futures[fut]
                review_results.append({
                    "idx": i, "dashboard": f"| {i} | 分析失败: {e} |",
                    "sections": [f"\n【候选 {i}】分析失败: {e}"],
                })

    # 按 idx 排序保持顺序
    review_results.sort(key=lambda r: r["idx"])
    review_sections = []
    dashboard_rows  = []
    for r in review_results:
        dashboard_rows.append(r["dashboard"])
        review_sections.extend(r["sections"])

    api_summary = (
        f"[API调用统计] 第一阶段: 0次（纯本地DB）  "
        f"第二阶段: 对 {actual_review} 只候选股并行调用 10~14 次（全维度分析，K线从本地DB读取，上限{MAX_REVIEW}只）  "
        f"第一阶段通过质量门槛({quality_threshold:.1f})的候选股: {qualified_count} 只"
    )

    _last_td = last_trade_day()
    _next_td = next_trade_day()
    pipeline_tag = "v2(z-score)" if use_v2_pipeline else "v1(legacy)"
    lines = [
        f"Full Stock Selection — {strategy_label} 策略  [pipeline: {pipeline_tag}]",
        "=" * 72,
        f"报告时间: {cn_now().strftime('%Y-%m-%d %H:%M')}  "
        f"数据截止: {_last_td}  "
        f"下一个交易日: {_next_td}",
        api_summary,
        "",
    ]

    # ═══ 数据健康告警 ═══
    # pct_chg 覆盖率或快照过期时, 直接把警告拍在报告开头. Staleness 以
    # last_trade_date (stock_history 最大日期) 为准, 不是 fundamentals.updated_at
    # 墙钟戳 — 之前那样算每周一都会误报 stale.
    if data_health and data_health.get("is_stale"):
        lines.extend([
            "⚠️  数据健康告警",
            f"  pct_chg覆盖率: {data_health.get('pct_chg_coverage', 0)*100:.1f}%   "
            f"最近交易日: {data_health.get('last_trade_date')}   "
            f"距今: {data_health.get('staleness_days', '?')} 天",
        ])
        for w in data_health.get("warnings", []):
            lines.append(f"  • {w}")
        lines.append("")

    lines.extend([
        "【1】环境感知层",
        market_report,
        "",
        "▌ 市场新闻面",
        news_report if news_report else "  （未获取新闻）",
        "",
        "▌ 北向资金（沪深港通）" + (f"  情绪修正: {northbound_adj:+.2f}" if northbound_adj else ""),
        northbound_report,
        "",
    ])

    # ═══ Sector Rotation Context ═══
    try:
        import sector_rotation as sr
        rotation = sr.sector_rotation_signal()
        if rotation:
            lines.extend([
                "▌ 行业轮动信号",
                f"  轮动阶段: {rotation.get('rotation_phase', 'N/A')}",
            ])
            overweight = rotation.get('overweight', [])[:3]
            if overweight:
                ow_text = ", ".join(s['sector'] for s in overweight)
                lines.append(f"  推荐超配: {ow_text}")
            underweight = rotation.get('underweight', [])[:3]
            if underweight:
                uw_text = ", ".join(s['sector'] for s in underweight)
                lines.append(f"  推荐低配: {uw_text}")
            lines.append("")
    except Exception:
        pass

    lines.extend([
        "【2】第一阶段候选池（本地DB，0次网络请求）",
        screen_report,
        "",
        "【3】深度复核仪表盘（前 {0} 只全维度分析）".format(actual_review),
        "| # | 股票 | 短/长(档) | 决策 | 仓位 | P(上涨) | 实时价格 |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    lines.extend(dashboard_rows)
    lines.extend([
        "",
        f"【4】第二阶段深度复核详情（前 {actual_review} 只）",
    ])
    lines.extend(review_sections)
    return "\n".join(lines)


# =====================================================
# Tool 4: Real-time Quote
# =====================================================

@mcp.tool()
def realtime_quote(stock_codes: str) -> str:
    """
    Get real-time quotes for one or more A-share stocks (latest price, change %, volume, etc.).

    Parameters
    ----------
    stock_codes : str
        Single code (e.g. "600519") or comma-separated codes (e.g. "600519,000858,600036").
        Shanghai stocks start with 6, Shenzhen with 0/3. No sh/sz prefix needed.

    Returns
    -------
    Formatted quote text including: stock name, latest price, change %,
    change amount, open, high, low, prev close, volume (lots), turnover (10k CNY), time.
    """
    codes = [c.strip() for c in stock_codes.split(",") if c.strip()]
    try:
        df = df_mod.get_realtime(codes)
    except Exception as e:
        return f"ERROR: Failed to fetch quotes: {e}"

    lines = ["Real-time Quotes", "-" * 50]
    for _, row in df.iterrows():
        arrow = "UP" if row["pct_chg"] >= 0 else "DOWN"
        lines.append(
            f"[{row['name']}] {row['code']}\n"
            f"  Latest: {row['current']:.2f}   "
            f"{arrow} {abs(row['pct_chg']):.2f}%  ({row['change']:+.2f})\n"
            f"  Open: {row['open']:.2f}   High: {row['high']:.2f}   Low: {row['low']:.2f}   Prev Close: {row['prev_close']:.2f}\n"
            f"  Volume: {row['volume']:.0f} lots   Turnover: {row['amount']:.1f} (10k CNY)\n"
            f"  Time: {row['date']} {row['time']}"
        )
    return "\n".join(lines)


# =====================================================
# Tool 2: K-line Data
# =====================================================

@mcp.tool()
def kline_data(
    stock_code: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    recent_days: int = 60,
    adjust: str = "none",
    with_indicators: bool = True,
) -> str:
    """
    Get A-share historical K-line data with optional technical indicators (MA/MACD/RSI/BOLL/KDJ).

    Parameters
    ----------
    stock_code      : str   Stock code, e.g. "600519" (Kweichow Moutai), "000858" (Wuliangye)
    period          : str   K-line period: daily / weekly / monthly
                            / 1m / 5m / 15m / 30m / 60m (intraday, trading days only)
    start_date      : str   Start date "YYYY-MM-DD", if empty uses recent_days
    end_date        : str   End date "YYYY-MM-DD", if empty uses today
    recent_days     : int   Effective when start_date is empty, fetch recent N days, default 60
    adjust          : str   Adjustment: none (unadjusted, default, matches quote software) /
                            qfq (forward, anchored to latest price, for indicators across splits) /
                            hfq (backward, rarely needed — upstream API currently behaves like none)
    with_indicators : bool  Whether to compute MA/MACD/RSI/BOLL/KDJ, default True

    Returns
    -------
    Text summary of the most recent 20 K-line bars + indicator values.
    """
    if not start_date and recent_days:
        start_date = (cn_now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")

    try:
        if period == "daily":
            df = df_mod.get_kline_prefer_db(
                stock_code,
                start=start_date or None,
                adjust=adjust,
            )
            if df is None or df.empty:
                return f"ERROR: No K-line data found for {stock_code}"
        else:
            df = df_mod.get_kline(
                stock_code,
                period=period,
                start=start_date or None,
                end=end_date or None,
                adjust=adjust,
            )
    except Exception as e:
        return f"ERROR: Failed to fetch K-line: {e}"

    name = df.attrs.get("name", stock_code)

    if with_indicators:
        df = df_mod.add_indicators(df)

    last = df.tail(20).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    lines = [
        f"{name} ({stock_code}) {period} K-line   Total {len(df)} bars",
        f"   Range: {df.iloc[0]['date'].strftime('%Y-%m-%d')} -> {df.iloc[-1]['date'].strftime('%Y-%m-%d')}",
        "-" * 70,
    ]

    base_cols = ["date", "open", "high", "low", "close", "pct_chg", "volume"]
    ind_cols  = ["ma5", "ma20", "macd", "rsi12", "kdj_k", "boll_ub", "boll_lb"]
    show_cols = base_cols + ([c for c in ind_cols if c in last.columns] if with_indicators else [])

    lines.append(last[show_cols].to_string(index=False))

    latest = df.iloc[-1]
    ma20_val = latest.get("ma20") if with_indicators else None
    rsi_val  = latest.get("rsi12") if with_indicators else None
    macd_val = latest.get("macd")  if with_indicators else None

    lines.append("-" * 70)
    lines.append(
        f"Latest Close: {latest['close']:.2f}  "
        f"Change: {latest['pct_chg']:+.2f}%"
    )
    if ma20_val:
        rel = "above" if latest["close"] > ma20_val else "below"
        lines.append(f"MA20: {ma20_val:.2f}  -> Price {rel} MA")
    if rsi_val is not None:
        zone = "overbought (>70)" if rsi_val > 70 else ("oversold (<30)" if rsi_val < 30 else "neutral")
        lines.append(f"RSI12: {rsi_val:.1f}  -> {zone}")
    if macd_val is not None:
        trend = "bullish" if macd_val > 0 else "bearish"
        lines.append(f"MACD Hist: {macd_val:.3f}  -> {trend} energy")

    return "\n".join(lines)


# =====================================================
# Tool 3: Financial Data
# =====================================================

@mcp.tool()
def financial_data(stock_code: str) -> str:
    """
    Get key financial and valuation indicators for an A-share stock (PE, PB, ROE, market cap, etc.).

    Parameters
    ----------
    stock_code : str   Stock code, e.g. "600519", "000858"

    Returns
    -------
    Includes PE TTM, PB, PS, total market cap, float market cap,
    ROE, gross margin, net margin, turnover rate, volume ratio, etc.
    """
    try:
        d = fin_mod.get_financial(stock_code)
    except Exception as e:
        return f"ERROR: Failed to fetch financial data: {e}"

    def fmt(v, suffix="", fmt=".2f"):
        if v is None:
            return "N/A"
        try:
            return f"{v:{fmt}}{suffix}"
        except Exception:
            return str(v)

    lines = [
        f"Financial Indicators  {d.get('name', stock_code)} ({stock_code})",
        "-" * 40,
        f"  Latest Price    : {fmt(d.get('current'))}",
        "",
        "  -- Valuation --",
        f"  PE TTM          : {fmt(d.get('pe_ttm'))}",
        f"  PE Static       : {fmt(d.get('pe_static'))}",
        f"  PB              : {fmt(d.get('pb'))}",
        f"  PS TTM          : {fmt(d.get('ps_ttm'))}",
        "",
        "  -- Market Cap --",
        f"  Total Mkt Cap   : {fmt(d.get('total_mv'), ' (100M CNY)')}",
        f"  Float Mkt Cap   : {fmt(d.get('float_mv'), ' (100M CNY)')}",
        "",
        "  -- Profitability --",
        f"  ROE             : {fmt(d.get('roe'), '%')}",
        f"  Gross Margin    : {fmt(d.get('gross_margin'), '%')}",
        f"  Net Margin      : {fmt(d.get('net_margin'), '%')}",
        "",
        "  -- Trading --",
        f"  Turnover Rate   : {fmt(d.get('turnover_rate'), '%')}",
        f"  Volume Ratio    : {fmt(d.get('volume_ratio'))}",
    ]
    return "\n".join(lines)


# =====================================================
# Tool 5: Batch Quote Overview
# =====================================================

@mcp.tool()
def batch_quote(stock_list: str) -> str:
    """
    Query multiple stocks at once for a concise overview, suitable for quick comparison.

    Parameters
    ----------
    stock_list : str
        Comma-separated stock codes, e.g. "600519,000858,600036,000001,300750"

    Returns
    -------
    Table-format multi-stock comparison sorted by change % descending.
    """
    codes = [c.strip() for c in stock_list.split(",") if c.strip()]
    if not codes:
        return "ERROR: Please enter at least one stock code, separated by commas"

    try:
        df = df_mod.get_realtime(codes)
    except Exception as e:
        return f"ERROR: Failed to fetch quotes: {e}"

    df = df.sort_values("pct_chg", ascending=False).reset_index(drop=True)

    warning = ""
    if len(df) < len(codes):
        failed = set(codes) - set(df["code"].tolist())
        warning = f"  WARNING: {len(failed)} code(s) returned no data: {', '.join(sorted(failed))}\n"

    lines = ["Batch Quote Overview", f"  Total {len(df)} stocks   {df.iloc[0]['date']} {df.iloc[0]['time']}",
             "-" * 65,
             f"  {'Name':<8} {'Code':<8} {'Latest':>8} {'Chg%':>8} {'Change':>7} {'Vol(lots)':>12} {'Turnover(10k)':>12}",
             "-" * 65]

    for _, row in df.iterrows():
        sign = "+" if row["pct_chg"] >= 0 else "-"
        lines.append(
            f"  {row['name']:<8} {row['code']:<8} "
            f"{row['current']:>8.2f} "
            f"{sign}{abs(row['pct_chg']):>6.2f}% "
            f"{row['change']:>+7.2f} "
            f"{row['volume']:>12,.0f} "
            f"{row['amount']:>12,.1f}"
        )

    lines.append("-" * 65)
    up   = (df["pct_chg"] > 0).sum()
    down = (df["pct_chg"] < 0).sum()
    flat = len(df) - up - down
    lines.append(f"  Up {up} / Flat {flat} / Down {down}")
    result = "\n".join(lines)
    if warning:
        result = warning + result
    return result


# =====================================================
# Tool 6: Stock Comprehensive Diagnosis (Core Tool)
# =====================================================

@mcp.tool()
def stock_diagnosis(
    stock_code: str,
    analysis_days: int = 250,
    news_sentiment: float = 0.0,
    monte_carlo_days: int = 20,
) -> str:
    """
    Perform full-dimensional quantitative diagnosis on an A-share stock (core tool).

    Runs 7 mathematical models simultaneously:
      Trend analysis (linear regression + ADX + MA alignment)
      Momentum analysis (MACD + RSI + ROC + Williams %R)
      Volatility analysis (ATR + Bollinger + historical volatility)
      Mean reversion (Z-Score multi-period deviation)
      Volume-price analysis (OBV + volume-price divergence)
      Support/Resistance (local extrema + Fibonacci retracement)
      Statistical features (Hurst exponent + skewness + kurtosis + Sharpe ratio)

    Outputs a composite score from -100 to +100 via weighted multi-factor scoring,
    along with a clear buy/hold/sell signal and confidence level.

    Also runs Monte Carlo simulation (5000 GBM paths) to provide
    future N-day up/down probabilities and price ranges.

    Parameters
    ----------
    stock_code       : str    Stock code, e.g. "600519"
    analysis_days    : int    Historical data days for analysis (default 250 ~ 1 year)
    news_sentiment   : float  News sentiment score assessed by you (Claude),
                              range -1.0 (extremely bearish) to +1.0 (extremely bullish),
                              0 means no news info or neutral
    monte_carlo_days : int    Monte Carlo simulation future days (default 20 trading days ~ 1 month)

    Returns
    -------
    Detailed diagnosis report with composite score, per-dimension scores,
    Monte Carlo probabilities, support/resistance levels, and risk warnings.
    """
    start = (cn_now() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    data_source = "db_prefer"
    try:
        df = df_mod.get_kline_prefer_db(stock_code, start=start, adjust="qfq")
        if df is None or df.empty:
            return f"ERROR: Failed to fetch K-line data for {stock_code}"
    except Exception as e:
        return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 30:
        return f"ERROR: Insufficient data: only {len(df)} bars, at least 30 required"

    # 尝试加载沪深300作为基准 (用于Beta/相对强弱计算)
    benchmark_close = None
    try:
        bm_df = sf.load_stock_history("000300")  # 沪深300指数
        if not bm_df.empty and len(bm_df) >= 60:
            benchmark_close = bm_df["close"].values
    except Exception:
        pass

    try:
        result = qe.comprehensive_diagnosis(
            df,
            news_sentiment=news_sentiment,
            run_monte_carlo=True,
            mc_days=monte_carlo_days,
            benchmark_close=benchmark_close,
        )
    except Exception as e:
        return f"ERROR: Diagnosis analysis failed: {e}"

    market_regime = sf.assess_market_regime()
    event_risk = sf.assess_event_risk(stock_code)
    decision = sf._build_trade_decision(
        df,
        result,
        float(df.iloc[-1]["close"]),
        market_regime,
        event_risk,
    )

    # Build detailed output
    lines = [result["summary"], f"Data Source: {data_source}"]
    lines.extend([
        "",
        "=" * 55,
        "Trade Decision:",
        f"  Conclusion      : {decision['conclusion']} ({decision['strength']}, confidence {decision['confidence']})",
        f"  Market Regime   : {market_regime['score']:+.1f}  {market_regime['label']} / {market_regime['action_bias']}",
        f"  Event Factor    : {event_risk['score']:+.1f}  {event_risk['label']}",
        f"  Position Advice : {decision['position_advice']}",
        f"  Holding Period  : {decision['holding_period']}",
        f"  Entry Style     : {decision['entry_style']}",
        f"  Entry Zone      : {decision['entry_low']:.2f} ~ {decision['entry_high']:.2f}" if decision.get('entry_low') is not None else "  Entry Zone      : N/A",
        f"  Stop Loss       : {decision['stop_loss']:.2f}" if decision.get("stop_loss") is not None else "  Stop Loss       : N/A",
        f"  Take Profit     : {decision['take_profit']:.2f}" if decision.get("take_profit") is not None else "  Take Profit     : N/A",
        f"  Risk/Reward     : 1:{decision['risk_reward']:.2f}" if decision.get("risk_reward") is not None else "  Risk/Reward     : N/A",
        f"  Kelly Position  : {decision['kelly_pct']:.1f}%" if decision.get("kelly_pct") is not None else "  Kelly Position  : N/A",
        f"  VaR(95)         : {decision['var_95_pct']:.2f}%" if decision.get("var_95_pct") is not None else "  VaR(95)         : N/A",
        f"  Event Summary   : {event_risk['summary']}",
    ])

    if decision.get("reasons_for"):
        lines.append("  Why this conclusion:")
        for item in decision["reasons_for"]:
            lines.append(f"    + {item}")

    if decision.get("reasons_against"):
        lines.append("  What is still missing / risky:")
        for item in decision["reasons_against"]:
            lines.append(f"    - {item}")

    if event_risk.get("positive_flags") or event_risk.get("negative_flags"):
        lines.append("  Event details:")
        for item in event_risk.get("positive_flags", [])[:3]:
            lines.append(f"    + {item}")
        for item in event_risk.get("negative_flags", [])[:3]:
            lines.append(f"    - {item}")

    lines.extend([
        f"  Invalidation    : {decision['invalidation']}",
        f"  Why not opposite: {decision['why_not_opposite']}",
    ])

    # Per-dimension details
    lines.extend([
        "",
        "=" * 55,
        "Per-Dimension Scores:",
    ])
    for key, factor in result["factors"].items():
        bar_len = int(abs(factor["score"]) / 5)
        bar = "#" * bar_len
        sign = "+" if factor["score"] >= 0 else ""
        indicator = "+" if factor["score"] > 0 else ("-" if factor["score"] < 0 else "=")
        lines.append(
            f"  {factor['label']:<18} : {sign}{factor['score']:>6.1f}  "
            f"(weight {factor['weight']:.0%})  {indicator} {bar}"
        )

    # Hurst exponent interpretation
    stats_detail = result.get("statistics_detail", {}).get("details", {})
    if "hurst" in stats_detail:
        lines.append(f"\n  Hurst Exponent: {stats_detail['hurst']:.3f} -> {stats_detail.get('hurst_interp', '')}")

    # ═════ ML Prediction Section (GradientBoosting) ═════
    try:
        import ml_predictor as mlp
        ml_result = mlp.predict_stock(stock_code, df, horizon=5)
        if ml_result and 'error' not in ml_result:
            lines.extend([
                "",
                "=" * 55,
                "ML Prediction (GradientBoosting):",
                f"  Probability Up (5d): {ml_result['prob_up']:.1f}%",
                f"  Probability Strong Up: {ml_result.get('prob_strong_up', 0):.1f}%",
                f"  Predicted Signal: {ml_result.get('signal', 'N/A')}",
                f"  Model Confidence: {ml_result.get('confidence', 0):.1f}%",
                f"  Validation AUC: {ml_result.get('validation_auc', 0):.3f}",
            ])
            if ml_result.get('feature_importance'):
                lines.append("  Top Features:")
                for fname, fimp in list(ml_result['feature_importance'].items())[:5]:
                    lines.append(f"    {fname}: {fimp:.3f}")
    except Exception:
        pass

    # ═════ Factor Exposure Section ═════
    try:
        import factor_model as fm
        fundamentals = {}
        try:
            overview = sf.load_stock_overview(stock_code)
            if overview:
                fundamentals = overview
        except Exception:
            pass

        factors_raw = fm.compute_single_stock_factors(
            stock_code, df, fundamentals,
            benchmark_close=benchmark_close
        )
        if factors_raw:
            lines.extend([
                "",
                "=" * 55,
                "Factor Exposure Analysis:",
            ])
            # Show top positive and negative factor exposures
            sorted_factors = sorted(factors_raw.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
            for fname, fval in sorted_factors[:8]:
                if fval is not None:
                    direction = "+" if fval > 0 else ""
                    lines.append(f"  {fname:<20}: {direction}{fval:.2f}")
    except Exception:
        pass

    return "\n".join(lines)


# =====================================================
# Tool 7: Strategy Backtest
# =====================================================

@mcp.tool()
def strategy_backtest(
    stock_code: str,
    strategy_name: str = "ma_cross",
    backtest_days: int = 500,
    initial_capital: float = 100000,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
) -> str:
    """
    Run historical strategy backtest on an A-share stock, simulating real trading
    (with commission, stamp tax, slippage).

    Parameters
    ----------
    stock_code      : str    Stock code, e.g. "600519"
    strategy_name   : str    Strategy name, options:
                             - "ma_cross"    MA Crossover (MA5/MA20 golden/death cross)
                             - "macd"        MACD golden/death cross
                             - "rsi"         RSI overbought/oversold reversal
                             - "bollinger"   Bollinger Band mean reversion
                             - "multifactor" Multi-factor composite scoring (slower)
    backtest_days   : int    Historical days for backtest (default 500 ~ 2 years)
    initial_capital : float  Starting capital (default 100,000 CNY)
    stop_loss_pct   : float  Stop-loss percentage (e.g. 0.05 = stop at 5% loss, 0 = disabled)
    take_profit_pct : float  Take-profit percentage (e.g. 0.10 = take at 10% gain, 0 = disabled)

    Returns
    -------
    Full backtest report: CAGR, Sharpe ratio, max drawdown, win rate, profit factor, trade log, etc.
    """
    start = (cn_now() - timedelta(days=backtest_days)).strftime("%Y-%m-%d")

    try:
        df = df_mod.get_kline_prefer_db(stock_code, start=start, adjust="qfq")
        if df is None or df.empty:
            return f"ERROR: Failed to fetch K-line data for {stock_code}"
    except Exception as e:
        return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 60:
        return f"ERROR: Insufficient data: only {len(df)} bars, backtest requires at least 60"

    # Resolve stock code to get canonical code and name for A-share-specific features
    try:
        resolved = resolve_stock(stock_code)
        if not resolved or "error" in resolved:
            # Fallback: use provided code without name
            canonical_code = stock_code
            stock_name = ""
        else:
            canonical_code = resolved.get("code", stock_code)
            stock_name = resolved.get("name", "")
    except Exception:
        canonical_code = stock_code
        stock_name = ""

    available = list(bt.STRATEGIES.keys())
    if strategy_name not in available:
        return f"ERROR: Unknown strategy: {strategy_name}. Available: {available}"

    try:
        result = bt.backtest(
            df,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct,
            code=canonical_code,           # NEW: Pass stock code for price limit detection
            name=stock_name,               # NEW: Pass stock name for ST detection
        )
    except Exception as e:
        return f"ERROR: Backtest execution failed: {e}"

    return bt.format_backtest_result(result)


# =====================================================
# Tool 8: Risk Assessment
# =====================================================

@mcp.tool()
def risk_assessment(
    stock_code: str,
    position_value: float = 100000,
    analysis_days: int = 250,
) -> str:
    """
    Perform comprehensive risk assessment on an A-share stock.

    Analysis includes:
      VaR (Value at Risk) -- 95% and 99% confidence
      CVaR (Conditional VaR) -- expected loss in extreme scenarios
      Kelly Criterion -- optimal position size recommendation
      ATR Dynamic Stops -- volatility-based stop-loss/take-profit levels
      Stress Test -- historical extreme replay + Monte Carlo extreme paths

    Parameters
    ----------
    stock_code     : str    Stock code, e.g. "600519"
    position_value : float  Current or planned position value (default 100,000 CNY)
    analysis_days  : int    Historical data days for analysis (default 250 ~ 1 year)

    Returns
    -------
    Complete risk report with VaR/CVaR, Kelly position, stop/take-profit levels,
    and stress test scenarios.
    """
    start = (cn_now() - timedelta(days=analysis_days)).strftime("%Y-%m-%d")

    try:
        df = df_mod.get_kline_prefer_db(stock_code, start=start, adjust="qfq")
        if df is None or df.empty:
            return f"ERROR: Failed to fetch K-line data for {stock_code}"
    except Exception as e:
        return f"ERROR: Failed to fetch K-line data: {e}"

    if len(df) < 30:
        return f"ERROR: Insufficient data: only {len(df)} bars"

    try:
        report = rm.comprehensive_risk_assessment(df, position_value=position_value)
    except Exception as e:
        return f"ERROR: Risk assessment failed: {e}"

    return report


# =====================================================
# Tool 9: Stock Screener
# =====================================================

@mcp.tool()
def stock_screener(
    strategy: str = "value",
    top_n: int = 20,
    deep_scan: bool = True,
    pe_max: float = 0,
    pe_min: float = 0,
    pb_max: float = 0,
    pb_min: float = 0,
    mv_min: float = 0,
    pct_chg_min: float = 0,
    pct_chg_max: float = 0,
    price_min: float = 0,
    price_max: float = 0,
) -> str:
    """
    ⚠️ 这是底层粗筛工具，不是选股入口！
    普通选股请求请使用 full_stock_selection，它包含完整的五面分析+北向资金+深度复核。
    只有用户明确要求"只看技术面粗筛、不需要新闻/财务/资金/风控分析"时才单独调用本工具。

    功能: 从本地SQLite扫描全市场约5000只A股，按技术面条件做初步粗筛。
    本工具不包含: 北向资金、新闻面、财务分析、估值分析、资金面、风控评估等。

    ★ API调用约束:
      本工具全部从本地 SQLite 读取，0 次外部网络请求。

    三阶段粗筛流程:
      Stage 0 历史横截面评分  — 读 SQLite stock_history，0次网络
      Stage 1 快速过滤        — 基于 stocks+fundamentals 内存操作，0次网络
      Stage 2 深度量化扫描    — 读 SQLite kline 做7维技术分析+蒙特卡洛，0次网络

    Parameters
    ----------
    strategy    : str   Screening strategy:
                        - "value"     Low PE + Low PB + decent market cap (classic value investing)
                        - "momentum"  Strong price trend + volume expansion (trend following)
                        - "oversold"  Deeply oversold stocks near support (mean reversion bounce)
                        - "potential" Fallen 10%+ in 60 days but fundamentals intact — find
                                      low-position stocks with reversal potential (bottom-fishing)
                        - "custom"    User-defined conditions (use the filter parameters below)
    top_n       : int   Number of top results to return (default 20)
    deep_scan   : bool  Whether to run Stage 2 deep quant analysis (default True).
                        Set to False for a faster scan using only basic metrics.
    pe_max      : float [custom] Max PE TTM (0 = no limit)
    pe_min      : float [custom] Min PE TTM (0 = no limit)
    pb_max      : float [custom] Max PB ratio (0 = no limit)
    pb_min      : float [custom] Min PB ratio (0 = no limit)
    mv_min      : float [custom] Min total market cap in 100M CNY (0 = no limit)
    pct_chg_min : float [custom] Min daily change % (0 = no limit)
    pct_chg_max : float [custom] Max daily change % (0 = no limit)
    price_min   : float [custom] Min stock price in CNY (0 = no limit)
    price_max   : float [custom] Max stock price in CNY (0 = no limit)

    Returns
    -------
    Screening report with ranked stock list, scores, signals, and key metrics.
    For deep scan: includes quant scores, trend direction, Monte Carlo probabilities.
    For fast scan: includes PE, PB, market cap, and fast filter scores.
    """
    # Build custom filter kwargs (only pass non-zero values)
    filter_kwargs = {}
    if strategy == "custom":
        if pe_max > 0:
            filter_kwargs["pe_max"] = pe_max
        if pe_min > 0:
            filter_kwargs["pe_min"] = pe_min
        if pb_max > 0:
            filter_kwargs["pb_max"] = pb_max
        if pb_min > 0:
            filter_kwargs["pb_min"] = pb_min
        if mv_min > 0:
            filter_kwargs["mv_min"] = mv_min
        if pct_chg_min != 0:
            filter_kwargs["pct_chg_min"] = pct_chg_min
        if pct_chg_max != 0:
            filter_kwargs["pct_chg_max"] = pct_chg_max
        if price_min > 0:
            filter_kwargs["price_min"] = price_min
        if price_max > 0:
            filter_kwargs["price_max"] = price_max

    try:
        result = sc.screen_stocks(
            strategy=strategy,
            top_n=top_n,
            deep_scan_enabled=deep_scan,
            stage1_limit=min(top_n * 4, 60),   # 硬上限60，防止Stage2过大
            **filter_kwargs,
        )
    except Exception as e:
        return f"ERROR: Stock screening failed: {e}"

    if "error" in result:
        return f"ERROR: {result['error']}"

    if not deep_scan:
        return result["summary"]

    lines = [result["summary"]]
    market_regime = result.get("market_regime")
    picks = result.get("results", [])

    if market_regime:
        lines.extend([
            "",
            "=" * 55,
            "Screening Decision Context:",
            f"  Market Regime : {market_regime['score']:+.1f}  {market_regime['label']} / {market_regime['action_bias']}",
            f"  Breadth       : Up {market_regime.get('up_ratio', 'N/A')}% / Down {market_regime.get('down_ratio', 'N/A')}%",
            f"  Avg Change    : {market_regime.get('avg_pct_chg', 'N/A')}%",
            f"  Interpretation: {market_regime['summary']}",
        ])

    if picks:
        lines.extend([
            "",
            "Top Actionable Picks:",
        ])
        for idx, item in enumerate(picks[:min(5, len(picks))], 1):
            decision = item.get("decision", {}) or {}
            lines.extend([
                f"{idx}. {item['name']} ({item['code']})  {decision.get('conclusion', item.get('signal', 'N/A'))}",
                f"   Price {item['current']:.2f}   EQS {item.get('entry_quality_score', 0):+.1f}   Quant {item.get('total_score', 0):+.1f}",
                f"   Position {decision.get('position_advice', 'N/A')}   Holding {decision.get('holding_period', 'N/A')}",
                f"   Entry {item.get('entry_low', 'N/A')} ~ {item.get('entry_high', 'N/A')}   Stop {item.get('stop_loss', 'N/A')}   TP {item.get('take_profit', 'N/A')}",
                f"   P(up) {item.get('prob_up', 'N/A')}   Risk/Reward {item.get('risk_reward', 'N/A')}   Event {item.get('event_score', 'N/A')}",
            ])
            for reason in decision.get("reasons_for", [])[:2]:
                lines.append(f"   + {reason}")
            for reason in decision.get("reasons_against", [])[:2]:
                lines.append(f"   - {reason}")
            event_risk = item.get("event_risk", {}) or {}
            for reason in event_risk.get("negative_flags", [])[:1]:
                lines.append(f"   - Event: {reason}")

    return "\n".join(lines)


# =====================================================
# Tool 10: 个股新闻面（新增）
# =====================================================

@mcp.tool()
def stock_news(stock_query: str, max_items: int = 8) -> str:
    """
    Analyze stock-specific news flow beyond market-wide headlines.

    This tool focuses on company-level and stock-specific public news, and should
    be used together with stock_announcements:
      - stock_news: company / media / industry-related news flow
      - stock_announcements: exchange announcements / filings / earnings windows

    Parameters
    ----------
    stock_query : str
        Stock code or stock name, e.g. "000066" or "中国长城"
    max_items : int
        Max number of stock-specific news headlines to show.

    Returns
    -------
    A Chinese report containing stock-specific headlines and a small
    news_sentiment delta for downstream diagnosis.
    """
    resolved = _resolve_stock_unique(stock_query)
    rows = resolved["rows"]
    if not rows:
        return (
            f"ERROR: No stock matched '{stock_query}' in local stocks table.\n"
            "Please update stocks first or provide a more precise code/name."
        )
    if len(rows) != 1:
        lines = [
            "ERROR: Stock query is ambiguous.",
            f"Query: {stock_query}",
            "Candidates:",
        ]
        for code, name, suspended in rows[:10]:
            tag = "停牌" if suspended else "正常"
            lines.append(f"- {code} {name} ({tag})")
        lines.append("Please retry with the exact code.")
        return "\n".join(lines)

    try:
        return na.get_stock_news_report(
            stock_name=resolved["name"],
            stock_code=resolved["code"],
            max_items=max_items,
        )
    except Exception as e:
        return f"ERROR: 个股新闻面获取失败: {e}"


# =====================================================
# Tool 11: 财报深度分析（新增）
# =====================================================

@mcp.tool()
def earnings_analysis(
    stock_code: str,
    periods: int = 8,
) -> str:
    """
    对某只A股进行财报深度分析，结合历史多期财务数据和历次财报公告的市场反应。

    【何时调用】
      - 当 stock_announcements 发现近期有年报/季报/业绩预告时，必须紧接着调用此工具
      - 当用户明确关心财报质量、业绩趋势、或财报是否超预期时

    分析内容：
      ① 历史财务趋势（最近 N 期）
           营业收入 / 同比增长率
           归母净利润 / 同比增长率
           毛利率、ROE 趋势
      ② 历次财报公告的市场反应
           公告当日涨跌幅 + 成交量倍数
           公告前5日 vs 后3日累计涨跌
           市场反应定性（强烈正面/正面/中性/负面/强烈负面）
      ③ 对当前分析的影响
           财报超预期 → news_sentiment +0.2~+0.3
           符合预期   → news_sentiment 不调整
           低于预期   → news_sentiment -0.2~-0.3

    Parameters
    ----------
    stock_code : str   股票代码，如 "000423"（东阿阿胶）
    periods    : int   分析历史期数，默认 8 期（约 2 年）

    Returns
    -------
    完整中文财报分析报告，含财务趋势表、历次市场反应、news_sentiment 调整建议。
    """
    # 获取股票名称
    try:
        rt = df_mod.get_realtime([stock_code])
        name = rt.iloc[0]["name"] if not rt.empty else stock_code
    except Exception:
        name = stock_code

    # ── 历史财务数据 ──
    try:
        history = fin_mod.get_financial_history(stock_code, periods=periods)
    except Exception as e:
        return f"ERROR: 历史财务数据获取失败: {e}"

    if not history:
        return "ERROR: 未获取到历史财务数据，请检查股票代码或稍后重试"

    # ── 公告列表，找财报类公告 ──
    try:
        anns = ann_mod.get_announcements(stock_code, page_size=30)
        earnings_anns = [a for a in anns if a["is_earnings"]][:6]
    except Exception:
        earnings_anns = []

    # ── 拉足够长的K线（一次请求复用于所有反应分析） ──
    kline_df = None
    if earnings_anns:
        try:
            oldest = earnings_anns[-1]["date"] if earnings_anns else ""
            start_dt = (
                (datetime.strptime(oldest, "%Y-%m-%d") - timedelta(days=40)).strftime("%Y-%m-%d")
                if oldest else
                (cn_now() - timedelta(days=600)).strftime("%Y-%m-%d")
            )
            kline_df = df_mod.get_kline(stock_code, period="daily",
                                    start=start_dt, adjust="none")
        except Exception:
            kline_df = None

    # ── 逐条分析财报市场反应 ──
    reactions = []
    for ann in earnings_anns:
        if ann["date"]:
            r = ann_mod.analyze_earnings_reaction(stock_code, ann["date"], kline_df)
            r["title"] = ann["title"]
            reactions.append(r)

    return ann_mod.format_earnings_analysis(stock_code, name, history, reactions)


# =====================================================
# Tool 12: 个股公告查询（新增）
# =====================================================

@mcp.tool()
def stock_announcements(
    stock_code: str,
    count: int = 20,
) -> str:
    """
    查询某只A股最近的公告列表，自动识别并高亮财报/业绩相关公告。

    【重要用途】在做股票分析之前，先调用此工具确认：
      - 是否已披露最新年报/季报/业绩预告
      - 财报披露日期（避免用旧财务数据做估值判断）
      - 是否有重大公告影响当前走势

    智能识别以下财报类公告并给出提示：
      年报 / 半年报 / 季报（一季报/三季报）/ 业绩预告 / 业绩快报 / 业绩修正

    Parameters
    ----------
    stock_code : str   股票代码，如 "000423"（东阿阿胶）
    count      : int   返回最近 N 条公告，默认 20

    Returns
    -------
    中文公告列表，包含：
      - 财报状态提示（是否已披露、披露日期）
      - 全部公告列表（财报类用★标注）
      - 财报数据是否可能尚未更新的风险提示
    """
    try:
        anns = ann_mod.get_announcements(stock_code, page_size=count)
    except Exception as e:
        return f"ERROR: 公告获取失败: {e}"

    # 尝试获取股票名称
    try:
        rt = df_mod.get_realtime([stock_code])
        name = rt.iloc[0]["name"] if not rt.empty else ""
    except Exception:
        name = ""

    return ann_mod.format_announcements(stock_code, anns, name=name)


# =====================================================
# Tool 13: 市场消息面环境日报（新增）
# =====================================================

@mcp.tool()
def market_news() -> str:
    """
    获取当日市场环境报告（外围行情 + 新闻面）。

    【建议在分析任何个股之前优先调用此工具】，以了解当日市场背景。

    内容涵盖:
      外围市场行情 —— 美股三大指数 + VIX、欧股、亚太、黄金/原油 (market_quote)
      国际/全球财经新闻 —— 东方财富全球市场 + 新浪财经 (仅保留2天内)
      国内A股/财经新闻 —— 东方财富要闻 + 股票新闻 (仅保留2天内)
      综合情绪研判 —— 外围行情评分 + 新闻关键词微调

    Returns
    -------
    完整中文市场环境报告（含行情表格 + 新闻列表 + 综合评分）。

    注意：本工具需要抓取网络数据，执行时间约 20-30 秒，请耐心等待。
    """
    try:
        sep = "═" * 56
        now = cn_now()
        lines = [
            sep,
            f"  市场环境报告   {now.strftime('%Y-%m-%d  %H:%M')}",
            sep,
            "",
        ]

        # ── 1. 外围行情 (market_quote) ──
        market_data = mq.get_foreign_markets()
        market_score = mq.compute_market_score(market_data)
        lines.append(mq.format_market_block(market_data))
        lines.append("")
        lines.append(mq.build_market_interpretation(market_data, market_score))

        # ── 2. 新闻面 (news_analyzer, 仅2天内) ──
        # 只调用一次 get_news_feeds()，避免重复网络请求
        lines.append("")
        intl_news, domestic_news = na.get_news_feeds()
        news_delta = na.news_sentiment_delta(intl_news, domestic_news)
        total_count = len(intl_news) + len(domestic_news)
        max_age = na._default_max_age_days()
        news_sep = "═" * 56
        news_lines = [
            news_sep,
            f"  市场新闻面报告   {now.strftime('%Y-%m-%d  %H:%M')}",
            f"  (时效窗口: {max_age}天内)",
            news_sep,
            "",
            na._fmt_news_block(intl_news, "【国际/全球财经新闻】"),
            "",
            na._fmt_news_block(domestic_news, "【国内A股/财经新闻】"),
            "",
            "【新闻面情绪结论】",
            f"  有效新闻数量: {total_count} 条（{max_age}天内）",
            f"  新闻情绪微调值: {news_delta:+.2f}  ({na._score_to_label(news_delta)})",
            news_sep,
        ]
        lines.append("\n".join(news_lines))

        # ── 3. 综合 ──
        final = round(max(-1.0, min(1.0, market_score + news_delta * 0.4)), 2)

        lines.extend([
            "",
            "【综合结论】",
            f"  外围行情评分:     {market_score:+.2f}  ({mq.score_to_label(market_score)})",
            f"  新闻面微调:       {news_delta:+.2f}",
            "  ─────────────────────────────",
            f"  综合建议 news_sentiment 参考值: {final:+.2f}",
            "",
            "  ➡  使用方法:",
            "     调用 stock_diagnosis 时，将此值传入 news_sentiment 参数。",
            "     Claude 可根据新闻实际内容在此基础上微调 ±0.1。",
            sep,
        ])
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: 市场环境报告获取失败: {e}"


# =====================================================
# Tool 14b: 北向资金成交活跃度
# =====================================================

@mcp.tool()
def northbound_flow(days: int = 10) -> str:
    """
    获取北向资金（沪深港通）成交活跃度数据。

    【背景】2024年5月起港交所不再披露北向资金净买入数据，
    本接口改为返回每日成交额和笔数，用于评估外资参与度趋势。

    【重要性】北向资金成交额反映外资对A股的参与热度：
      - 成交额持续放量 → 外资关注度上升，市场活跃度增加
      - 成交额持续缩量 → 外资兴趣减退，参与度下降
      - 需结合市场涨跌方向综合判断（放量+涨=看多，放量+跌=分歧加大）

    【调用时机】
      - market_news 之后立即调用，用于修正 news_sentiment
      - 选股流程中评估外资参与热度
      - 单股分析中判断市场整体活跃度

    【news_sentiment 调整规则（基于成交额趋势）】
      近5日成交额变化 > +20%  → +0.10（外资关注度显著提升）
      近5日成交额变化 +5~20%  → +0.05（温和放量）
      近5日成交额变化 -5~+5%  → 不调整（持平）
      近5日成交额变化 -20~-5% → -0.05（温和缩量）
      近5日成交额变化 < -20%  → -0.10（大幅缩量，兴趣减退）
      连续5日以上缩量         → 建议降低仓位积极性

    Parameters
    ----------
    days : int   返回最近N个交易日数据，默认10日

    Returns
    -------
    中文报告，含每日沪/深股通成交额、合计、趋势判断、news_sentiment 调整建议。
    """
    try:
        flow = cf_mod.get_northbound_flow(days=days)
    except Exception as e:
        return f"ERROR: 北向资金数据获取失败: {e}"

    if not flow:
        return "ERROR: 未获取到北向资金数据，请稍后重试"

    report = cf_mod.format_northbound_report(flow)

    # 追加 news_sentiment 调整建议（基于成交额趋势）
    amts = [d.get("total_deal_amt", 0) for d in flow]
    recent5 = amts[:5]
    early5 = amts[-5:] if len(amts) >= 10 else amts[:5]
    avg_recent = sum(recent5) / len(recent5) if recent5 else 0
    avg_early = sum(early5) / len(early5) if early5 else 0
    trend_pct = (avg_recent / avg_early - 1) * 100 if avg_early > 0 else 0

    # 连续缩量天数
    consecutive_shrink = 0
    for i in range(len(amts) - 1):
        if amts[i] < amts[i + 1]:
            consecutive_shrink += 1
        else:
            break

    if trend_pct > 20:
        sentiment_adj = "+0.10"
        advice = "外资成交大幅放量，关注度显著提升，偏积极"
    elif trend_pct > 5:
        sentiment_adj = "+0.05"
        advice = "外资成交温和放量，参与度上升"
    elif trend_pct > -5:
        sentiment_adj = "0"
        advice = "外资成交持平，不影响原有判断"
    elif trend_pct > -20:
        sentiment_adj = "-0.05"
        advice = "外资成交温和缩量，参与度下降"
    else:
        sentiment_adj = "-0.10"
        advice = "外资成交大幅缩量，建议降低仓位积极性"

    report += f"\n  ▶ 成交额趋势变化: {trend_pct:+.1f}%"
    report += f"\n  ▶ news_sentiment 调整建议: {sentiment_adj}"
    report += f"\n  ▶ 操作含义: {advice}"
    if consecutive_shrink >= 5:
        report += f"\n  ⚠️ 警告: 已连续 {consecutive_shrink} 日成交缩量，外资参与度持续下降，建议降低仓位积极性"

    return report


# =====================================================
# Tool 14: 量化资金活跃度分析（新增）
# =====================================================

@mcp.tool()
def quant_activity(
    stock_code: str,
    tick_n: int = 1500,
) -> str:
    """
    分析某只A股的量化资金参与程度（需在交易时段调用）。

    通过三个维度的实时公开数据推断量化资金活跃度:
      ① 均单手数     — 量化大量拆单 → 单笔成交手数极小（数据: 东方财富逐笔成交）
      ② 微单笔数占比 — 量化倾向≤5手微单 → 微单笔数占全部笔数的比例
      ③ 分钟成交均匀度 — 量化VWAP策略 → 全天成交量趋于均匀（CV低）

    综合输出 0-100 的量化活跃度得分:
      80-100: 量化主导  — 算法/高频资金为主要参与方
      62-79:  量化活跃  — 量化与散户/游资混合博弈
      42-61:  量化参与  — 有量化参与但非主导
      0-41:   量化偏少  — 散户/游资情绪交易为主

    并根据得分给出针对性的操作建议（止损策略/追涨策略/下单方式等）。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"
    tick_n     : int   分析最近 N 笔逐笔成交，默认 1500（越大越准确，但耗时稍长）

    Returns
    -------
    中文分析报告，包含三维度得分、买卖力道分析、针对性操作建议。

    注意: 逐笔数据仅在交易时段（09:30-15:00）有效，非交易时段调用会提示无数据。
    """
    try:
        return qd.get_quant_activity_report(stock_code, tick_n=tick_n)
    except Exception as e:
        return f"ERROR: 量化活跃度分析失败: {e}"


# =====================================================
# Tool 16: 主力资金流向（大单净流入）
# =====================================================

@mcp.tool()
def moneyflow(
    stock_code: str,
    days: int = 10,
) -> str:
    """
    查询个股主力资金流向（大单净流入/净卖出），输出信号解读报告。

    主力资金 = 单笔成交金额 ≥ 50万元的大单；
    超大单   = 单笔 ≥ 200万元。

    主力净流入连续多日 → 机构/大资金在建仓（看多信号）
    主力净流出加速    → 主力出货减仓（看空信号）

    建议在 stock_diagnosis 之前调用，用于补充技术面缺失的资金面视角。
    短线分析和选股复核时优先调用。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"
    days       : int   查询最近 N 个交易日，默认 10

    Returns
    -------
    中文主力资金报告（含逐日明细、近5日均值、信号判断）
    """
    try:
        data = cf_mod.get_moneyflow(stock_code, days=days)
        return cf_mod.format_moneyflow_report(data)
    except Exception as e:
        return f"ERROR: 主力资金流向获取失败: {e}"


# =====================================================
# Tool 17: 融资融券余额（两融）
# =====================================================

@mcp.tool()
def margin_trading(
    stock_code: str,
    days: int = 20,
) -> str:
    """
    查询个股融资融券余额历史数据（两融余额变化趋势）。

    融资余额上升 → 场内资金加杠杆做多，看多信号
    融资余额快速下降 → 多头去杠杆/止损平仓，谨慎信号
    融券余额大幅上升 → 机构/大户加大做空头寸，看空压力

    两融余额是评估场内多空力量对比的重要参考，
    尤其对于两融标的股票（沪深300成分股等），建议在 stock_diagnosis 之前调用。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"
    days       : int   查询最近 N 条记录，默认 20

    Returns
    -------
    中文两融余额报告（含逐日融资融券余额、趋势信号）
    若该股未纳入两融标的，会明确说明。
    """
    try:
        data = cf_mod.get_margin_trading(stock_code, days=days)
        return cf_mod.format_margin_report(data)
    except Exception as e:
        return f"ERROR: 两融余额数据获取失败（该股可能未纳入两融标的）: {e}"


# =====================================================
# Tool 18: 资产负债表关键指标（负债率/商誉）
# =====================================================

@mcp.tool()
def balance_sheet(
    stock_code: str,
) -> str:
    """
    查询个股最新资产负债表关键指标，包含高商誉预警。

    关键指标:
      资产负债率  — 越低越稳健，>70% 为高负债预警
      商誉/净资产 — >30% 为高商誉风险，商誉减值可能大幅侵蚀利润
      货币资金    — 衡量短期偿债能力

    重要用途:
      - 高商誉个股筛除（对应框架中"避开高商誉个股"）
      - 低负债率验证（长线选股的安全边际）
      - 现金充裕度评估（抗风险能力）

    建议在 financial_data 之后调用，补充资产负债健康度评估。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"

    Returns
    -------
    中文资产负债关键指标报告，含高商誉/高负债预警
    """
    try:
        d = fin_mod.get_balance_sheet(stock_code)
        lines = [
            f"资产负债表关键指标 — {stock_code}",
            f"  报告期:       {d.get('report_date', 'N/A')}",
            "=" * 50,
            f"  总资产:       {d.get('total_assets') or 'N/A'} 亿元",
            f"  总负债:       {d.get('total_liab') or 'N/A'} 亿元",
            f"  归母净资产:   {d.get('equity') or 'N/A'} 亿元",
            f"  货币资金:     {d.get('cash') or 'N/A'} 亿元",
            "-" * 50,
        ]

        # 资产负债率
        dr = d.get("debt_ratio")
        if dr is not None:
            dr_flag = "⚠️ 高负债风险" if dr > 70 else ("✅ 负债健康" if dr < 50 else "🟡 负债适中")
            lines.append(f"  资产负债率:   {dr:.1f}%  {dr_flag}")
        else:
            lines.append("  资产负债率:   N/A")

        # 商誉
        gw    = d.get("goodwill") or 0
        gw_r  = d.get("goodwill_ratio")
        if gw_r is not None:
            gw_flag = "🔴 高商誉警告！减值风险高" if gw_r > 30 else ("⚠️ 商誉偏高，需关注" if gw_r > 15 else "✅ 商誉占比低")
            lines.append(f"  商誉:         {gw:.2f} 亿元 (占净资产 {gw_r:.1f}%)  {gw_flag}")
        else:
            lines.append(f"  商誉:         {gw:.2f} 亿元")

        lines.extend([
            "",
            "注：高商誉（>30%净资产）个股在经济下行期面临大额减值风险，应谨慎介入。",
        ])
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: 资产负债表数据获取失败: {e}"


# =====================================================
# Tool 19: 分红历史
# =====================================================

@mcp.tool()
def dividend_history(
    stock_code: str,
    years: int = 5,
) -> str:
    """
    查询个股近几年的历史分红派息记录。

    持续分红是公司治理优良、现金流充裕的信号，
    是长线价值投资的重要参考维度（对应"分红历史"因素）。

    分析要点:
      - 连续分红年数和分红频率
      - 每股分红是否稳定或增长
      - 总分红金额变化趋势

    建议在 financial_data 和 balance_sheet 之后调用，
    用于长线选股或价值型策略复核。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"
    years      : int   查询最近 N 年记录，默认 5 年

    Returns
    -------
    中文分红历史报告
    """
    try:
        data = fin_mod.get_dividend_history(stock_code, years=years)
        if not data:
            return f"⚠️ {stock_code} 近{years}年无分红记录（或数据暂不可用）"

        lines = [
            f"分红历史记录 — {stock_code}",
            "=" * 55,
            f"  {'报告期':<12} {'公告日':<12} {'除权日':<12} {'每股分红':>8} {'送股比':>6} {'分红总额':>8}",
            "-" * 55,
        ]
        for d in data:
            dps = d.get("div_per_share")
            bon = d.get("bonus_ratio")
            tot = d.get("total_div")
            lines.append(
                f"  {d.get('report_date',''):<12} {d.get('ann_date',''):<12} "
                f"{d.get('ex_date',''):<12} "
                f"{(str(dps)+'元') if dps else '—':>8} "
                f"{(str(bon)+'股') if bon else '—':>6} "
                f"{(str(tot)+'亿') if tot else '—':>8}"
            )

        lines.append("-" * 55)

        # 统计分析
        divs = [d.get("div_per_share") for d in data if d.get("div_per_share") is not None and d.get("div_per_share") > 0]
        if divs:
            lines.extend([
                f"  共 {len(divs)} 次现金分红，平均每股 {sum(divs)/len(divs):.3f} 元",
                f"  最近一次每股: {divs[0]:.3f} 元 / 最高: {max(divs):.3f} 元",
                "",
                "  分红质量评估: " + (
                    "✅ 连续多年分红，股东回报稳健" if len(divs) >= 4 else
                    "🟡 分红记录有限，需关注现金流" if len(divs) >= 2 else
                    "🟠 分红次数较少"
                ),
            ])
        else:
            lines.append("  ⚠️ 无现金分红记录（该公司不分红或数据缺失）")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: 分红历史数据获取失败: {e}"


# =====================================================
# Tool 20: PEG + 杜邦分析（估值质量增强）
# =====================================================

@mcp.tool()
def valuation_quality(
    stock_code: str,
) -> str:
    """
    计算PEG比率 + 杜邦分析，评估估值质量与基本面竞争力。

    PEG (Price/Earnings to Growth):
      PEG = PE(TTM) / 净利润增速(%)
      PEG < 1  → 成长性充分支撑估值（价值洼地）
      PEG 1~2  → 合理区间
      PEG > 2  → 估值偏高，成长性不足

    杜邦分析:
      拆解ROE来源：ROE = 净利润率 × 资产周转率 × 权益乘数
      分析ROE趋势、毛利率趋势、净利润增速一致性
      判断盈利是否可持续

    建议在 financial_data 和 earnings_analysis 之后调用，
    用于完成估值质量闭环，特别适合价值型选股复核。

    Parameters
    ----------
    stock_code : str   股票代码，如 "600519"

    Returns
    -------
    中文PEG+杜邦分析报告，含估值结论
    """
    try:
        # 获取财务数据
        fin = fin_mod.get_financial(stock_code)
        hist = fin_mod.get_financial_history(stock_code, periods=8)

        pe_ttm = fin.get("pe_ttm")
        name   = fin.get("name", stock_code)

        lines = [
            f"估值质量分析 — {name}({stock_code})",
            "=" * 55,
        ]

        # PEG
        peg_result = fin_mod.compute_peg(hist, pe_ttm)
        lines.extend([
            "【PEG分析】",
            f"  当前PE(TTM):    {pe_ttm or 'N/A'}",
            f"  增速计算方式:   {peg_result.get('growth_method', 'N/A')}",
            f"  净利润增速:     {peg_result.get('growth_rate') or 'N/A'}%",
            f"  PEG值:          {peg_result.get('peg') or 'N/A'}",
            f"  PEG评估:        {peg_result.get('verdict', 'N/A')}",
            "",
        ])

        # 杜邦分析
        dupont = qe.analyze_dupont(hist)
        lines.extend([
            "【杜邦分析】",
            f"  近期平均ROE:    {dupont.get('avg_roe') or 'N/A'}%",
            f"  ROE趋势:        {dupont.get('roe_trend', 'N/A')}",
            f"  ROE质量来源:    {dupont.get('roe_quality', 'N/A')}",
            f"  毛利率趋势:     {dupont.get('gross_trend', 'N/A')}",
            f"  平均毛利率:     {dupont.get('avg_gross_margin') or 'N/A'}%",
            f"  净利润增速一致性: {dupont.get('profit_trend', 'N/A')}",
            f"  杜邦综合评分:   {dupont.get('score', 0)} 分",
            f"  综合判断:       {dupont.get('verdict', 'N/A')}",
            "",
        ])

        # 财务历史摘要
        if hist:
            lines.extend([
                "【近期财务趋势（最新4期）】",
                f"  {'报告期':<12} {'营收YoY':>8} {'净利YoY':>8} {'毛利率':>7} {'ROE':>7}",
                "-" * 50,
            ])
            for r in hist[:4]:
                lines.append(
                    f"  {r.get('report_date',''):<12} "
                    f"{(str(r.get('revenue_yoy'))+'%') if r.get('revenue_yoy') is not None else 'N/A':>8} "
                    f"{(str(r.get('profit_yoy'))+'%') if r.get('profit_yoy') is not None else 'N/A':>8} "
                    f"{(str(r.get('gross_margin'))+'%') if r.get('gross_margin') is not None else 'N/A':>7} "
                    f"{(str(r.get('roe'))+'%') if r.get('roe') is not None else 'N/A':>7}"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: 估值质量分析失败: {e}"


# =====================================================
# Tool 21: Portfolio Optimization
# =====================================================

@mcp.tool()
def portfolio_optimize(
    stock_codes: str,
    method: str = "max_sharpe",
    total_capital: float = 100000,
) -> str:
    """
    Portfolio optimization for a set of stocks.

    Args:
        stock_codes: Comma-separated stock codes (e.g., "600519,000858,601318")
        method: Optimization method - 'max_sharpe', 'min_variance', 'risk_parity', 'equal_weight'
        total_capital: Total investment capital in CNY

    Returns detailed portfolio allocation with weights, shares, risk decomposition.
    """
    try:
        import portfolio_optimizer as po

        codes = [c.strip() for c in stock_codes.split(",") if c.strip()]
        if not codes:
            return "ERROR: No valid stock codes provided"

        if method not in ("max_sharpe", "min_variance", "risk_parity", "equal_weight"):
            return f"ERROR: Unknown method '{method}'. Use: max_sharpe, min_variance, risk_parity, equal_weight"

        # Load histories for all stocks
        histories = {}
        prices = {}
        for code in codes:
            h = sf.load_stock_history(code)
            if not h.empty and len(h) >= 60:
                histories[code] = h.tail(250).reset_index(drop=True)
                prices[code] = float(h.iloc[-1]["close"])

        valid_codes = list(histories.keys())
        if len(valid_codes) < 2:
            return "ERROR: Need at least 2 stocks with sufficient history (≥60 bars)"

        lines = [
            f"Portfolio Optimization — {method.upper()}",
            "=" * 60,
            f"Stock Codes: {', '.join(valid_codes)}",
            f"Total Capital: {total_capital:,.0f} CNY",
            f"Optimization Method: {method}",
            "",
        ]

        # Estimate returns and covariance
        mu = po.estimate_returns(histories, method="shrinkage")
        cov = po.estimate_covariance(histories, method="ledoit_wolf")

        # Run optimization
        if method == "max_sharpe":
            result = po.optimize_max_sharpe(mu, cov)
        elif method == "min_variance":
            result = po.optimize_min_variance(cov)
        elif method == "risk_parity":
            result = po.optimize_risk_parity(cov)
        else:  # equal_weight
            result = po.optimize_equal_weight(valid_codes)

        if "error" in result:
            return f"ERROR: Optimization failed: {result['error']}"

        weights = result.get("weights", {})
        lines.extend([
            "【Allocation】",
            f"  {'Stock':<10} {'Weight':>8} {'Capital':>12} {'Shares':>10}",
            "-" * 42,
        ])

        for code in valid_codes:
            w = weights.get(code, 0)
            alloc = total_capital * w
            price = prices.get(code, 0)
            shares = int(alloc / (price * 100)) * 100 if price > 0 else 0  # A-share 100-lot
            lines.append(f"  {code:<10} {w:>7.1%} {alloc:>12,.0f} {shares:>10}")

        exp_ret = result.get("expected_return")
        exp_vol = result.get("volatility")
        sharpe = result.get("sharpe")
        lines.extend([
            "",
            "【Performance Metrics】",
            f"  Expected Annual Return: {exp_ret:.2%}" if exp_ret is not None else "  Expected Annual Return: N/A",
            f"  Expected Annual Volatility: {exp_vol:.2%}" if exp_vol is not None else "  Expected Annual Volatility: N/A",
            f"  Sharpe Ratio: {sharpe:.3f}" if sharpe is not None else "  Sharpe Ratio: N/A",
        ])

        # Risk decomposition
        try:
            decomp = po.portfolio_risk_decomposition(weights, cov)
            if decomp and "contributions" in decomp:
                lines.extend(["", "【Risk Contribution】"])
                for code, pct in decomp["contributions"].items():
                    lines.append(f"  {code:<10} {pct:>7.1%}")
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: Portfolio optimization failed: {e}"


# =====================================================
# Tool 22: Sector Analysis & Rotation
# =====================================================

@mcp.tool()
def sector_analysis() -> str:
    """
    Analyze sector rotation and momentum across all A-share sectors.

    Returns sector rankings, rotation signals, overweight/underweight recommendations,
    and current market cycle phase.
    """
    try:
        import sector_rotation as sr

        lines = [
            "Sector Rotation Analysis",
            "=" * 70,
            f"Report Time: {cn_now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Get sector rotation signal
        rotation = sr.sector_rotation_signal()
        if rotation:
            lines.extend([
                "【Rotation Phase】",
                f"  Phase: {rotation.get('rotation_phase', 'N/A')}",
                f"  Cycle Stage: {rotation.get('cycle_stage', 'N/A')}",
                "",
            ])

            # Overweight sectors
            overweight = rotation.get("overweight", [])
            if overweight:
                lines.append("【Overweight Sectors】")
                for s in overweight[:5]:
                    lines.append(
                        f"  {s['sector']:<15} Momentum: {s.get('momentum', 'N/A'):>6}  "
                        f"Return: {s.get('return', 'N/A'):>6}"
                    )
                lines.append("")

            # Underweight sectors
            underweight = rotation.get("underweight", [])
            if underweight:
                lines.append("【Underweight Sectors】")
                for s in underweight[:5]:
                    lines.append(
                        f"  {s['sector']:<15} Momentum: {s.get('momentum', 'N/A'):>6}  "
                        f"Return: {s.get('return', 'N/A'):>6}"
                    )
                lines.append("")

            # Sector rankings
            lines.append("【Sector Momentum Ranking (20d)】")
            rankings = sr.sector_momentum_ranking(days=20)
            lines.append(f"  {'Rank':<5} {'Sector':<15} {'Momentum':>10} {'Return':>8}")
            lines.append("-" * 40)
            for i, s in enumerate(rankings[:10], 1):
                lines.append(
                    f"  {i:<5} {s['sector']:<15} {s.get('momentum', 'N/A'):>10} "
                    f"{s.get('return', 'N/A'):>8}"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: Sector analysis failed: {e}"


# =====================================================
# Tool 23: Multi-Factor Analysis
# =====================================================

@mcp.tool()
def factor_analysis(stock_code: str = "") -> str:
    """
    Multi-factor analysis tool with two operating modes:

    Mode 1 — Single-stock factor exposure (stock_code provided, e.g. "600519"):
      Returns z-scored factor values (Value, Quality, Momentum, Volatility, Size, Liquidity,
      Leverage, Reversal) for that stock vs universe. Positive z > +1 = top quartile.

    Mode 2 — Cross-sectional rankings (stock_code="" or omitted):
      Runs composite_alpha_score across all stocks in the universe and returns top/bottom
      ranked stocks by alpha. Use this for stock screening or factor crowding checks.
      Factors exceeding 80% crowding trigger a WARNING in the output.

    Args:
        stock_code: Optional stock code (e.g. "600519"). Leave empty for cross-sectional view.

    Examples:
        factor_analysis("600519")  → factor exposure for Kweichow Moutai
        factor_analysis("")        → cross-sectional alpha rankings for all stocks
    """
    try:
        import factor_model as fm

        lines = [
            "Multi-Factor Analysis",
            "=" * 70,
            f"Report Time: {cn_now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if stock_code and stock_code.strip():
            # Single stock factor exposure
            stock_code = stock_code.strip()
            lines.extend([
                f"【Single Stock Factor Exposure】 {stock_code}",
                "-" * 70,
            ])

            try:
                df = sf.load_stock_history(stock_code)
                if df is None or df.empty or len(df) < 30:
                    return f"ERROR: Insufficient data for stock {stock_code}"

                fundamentals = {}
                try:
                    overview = sf.load_stock_overview(stock_code)
                    if overview:
                        fundamentals = overview
                except Exception:
                    pass

                # Load benchmark
                benchmark_close = None
                try:
                    bm_df = sf.load_stock_history("000300")
                    if not bm_df.empty and len(bm_df) >= 60:
                        benchmark_close = bm_df["close"].values
                except Exception:
                    pass

                factors = fm.compute_single_stock_factors(
                    stock_code, df, fundamentals, benchmark_close=benchmark_close
                )
                if factors:
                    lines.append(f"  {'Factor':<25} {'Exposure':>12} {'Score':>10}")
                    lines.append("-" * 50)
                    sorted_factors = sorted(
                        factors.items(),
                        key=lambda x: abs(x[1]) if x[1] is not None else 0,
                        reverse=True,
                    )
                    for fname, fval in sorted_factors[:15]:
                        if fval is not None:
                            direction = "+" if fval > 0 else ""
                            lines.append(f"  {fname:<25} {direction}{fval:>11.3f}")
                else:
                    lines.append("  No factor data available")
            except Exception as e:
                lines.append(f"  Error: {e}")

        else:
            # Cross-sectional factor rankings
            lines.append("【Cross-Sectional Factor Rankings】 (Top stocks)")
            lines.append("-" * 70)

            try:
                # Load all stocks and compute factors
                conn = sf._get_db()
                stocks = db.fetchall(conn,
                    "SELECT code, name FROM stocks WHERE suspended = 0 LIMIT 100")

                all_factors = []
                for code, name in stocks[:30]:  # Top 30 for speed
                    try:
                        df = sf.load_stock_history(code)
                        if df is None or df.empty or len(df) < 30:
                            continue
                        factors = fm.compute_single_stock_factors(code, df, {})
                        if factors:
                            all_factors.append({"code": code, "name": name, "factors": factors})
                    except Exception:
                        continue

                # Build sector map for neutralization
                sector_map = {}
                try:
                    sector_rows = db.fetchall(conn,
                        "SELECT code, sector FROM stocks WHERE suspended = 0")
                    sector_map = {r[0]: (r[1] or "unknown") for r in sector_rows}
                except Exception:
                    pass

                if all_factors:
                    # Apply sector neutralization (0.7 strength) to reduce hot-sector clustering
                    try:
                        import pandas as _pd
                        factor_df = _pd.DataFrame(
                            [item["factors"] for item in all_factors],
                            index=[item["code"] for item in all_factors],
                        )
                        if sector_map and not factor_df.empty:
                            factor_df = fm.sector_neutralize(factor_df, sector_map)
                            for item in all_factors:
                                if item["code"] in factor_df.index:
                                    item["factors"] = factor_df.loc[item["code"]].to_dict()
                    except Exception:
                        pass  # fallback: use raw factors without neutralization

                    # Show top factors
                    all_factor_names = set()
                    for item in all_factors:
                        all_factor_names.update(item["factors"].keys())

                    for factor_name in list(all_factor_names)[:5]:
                        lines.append(f"\n【{factor_name}】 Top 5 Stocks")
                        lines.append(f"  {'Code':<8} {'Name':<12} {'Value':>10}")
                        lines.append("-" * 32)

                        factor_data = [
                            (
                                item["code"],
                                item["name"],
                                item["factors"].get(factor_name),
                            )
                            for item in all_factors
                            if factor_name in item["factors"]
                        ]
                        factor_data.sort(key=lambda x: abs(x[2]) if x[2] is not None else 0, reverse=True)

                        for code, name, val in factor_data[:5]:
                            if val is not None:
                                direction = "+" if val > 0 else ""
                                lines.append(f"  {code:<8} {name:<12} {direction}{val:>9.2f}")
                else:
                    lines.append("  No factor data available")
            except Exception as e:
                lines.append(f"  Error computing cross-sectional factors: {e}")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: Factor analysis failed: {e}"


# =====================================================
# Tool: Institutional Holdings (机构持仓 + 十大流通股东)
# =====================================================

@mcp.tool()
def institutional_holdings(stock_code: str) -> str:
    """
    Query institutional holdings and top 10 circulating shareholders for an A-share stock.

    Returns:
    - Top 10 circulating shareholders (name, type, holding %, change direction)
    - Fund holdings (which mutual funds hold this stock, holding amount, % of NAV)
    - Institutional consensus score (0-100): fund count + smart money + holder changes

    Data source: AKShare (Sina Finance + East Money), updated quarterly.
    Use this tool when asked about: 机构持仓, 基金持仓, 十大股东, 十大流通股东,
    institutional holdings, fund holdings, major shareholders, smart money.

    Args:
        stock_code: Stock code (e.g. "600498") or name (e.g. "烽火通信")
    """
    try:
        resolved = _resolve_stock_unique(stock_code)
        rows = resolved.get("rows", [])
        if not rows:
            return (
                f"ERROR: No stock matched '{stock_code}' in local stocks table.\n"
                "Please update stocks first or provide a more precise code/name."
            )
        if len(rows) != 1 or not resolved.get("code"):
            lines = [
                "ERROR: Stock query is ambiguous.",
                f"Query: {stock_code}",
                "Candidates:",
            ]
            for c, n, s in rows[:10]:
                tag = "停牌" if s else "正常"
                lines.append(f"- {c} {n} ({tag})")
            lines.append("Please retry with the exact code.")
            return "\n".join(lines)
        code = resolved["code"]

        from institutional import (
            get_fund_holdings, get_top_holders,
            store_fund_holdings, store_top_holders,
            format_institutional_report,
        )

        errors = []

        # ── 1. 数据库优先: 只要 DB 里有任何数据 (基金 OR 十大股东), 就用 DB ──
        #    这是关键: 即使 akshare 实时拉取失败, DB 已经有昨天/上季度的数据也能输出
        try:
            _db_conn = sf._get_db()
            try:
                try:
                    db.init_schema(_db_conn)
                except Exception:
                    pass
                fund_cnt_row = db.fetchone(_db_conn,
                    "SELECT COUNT(*) FROM fund_holdings WHERE code = ?", (code,))
                holder_cnt_row = db.fetchone(_db_conn,
                    "SELECT COUNT(*) FROM stock_top_holders WHERE code = ?", (code,))
                db_fund_cnt = (fund_cnt_row[0] if fund_cnt_row else 0) or 0
                db_holder_cnt = (holder_cnt_row[0] if holder_cnt_row else 0) or 0
            finally:
                _db_conn.close()
        except Exception as e:
            db_fund_cnt = 0
            db_holder_cnt = 0
            errors.append(f"DB 查询: {e}")

        if db_fund_cnt > 0 or db_holder_cnt > 0:
            try:
                report = format_institutional_report(code)
                # 只要拿到了报告 (不是 "暂无当季..." 的空报告), 就直接返回
                # 如果是空报告, 说明当季没数据, 继续走实时拉取
                if "暂无当季" not in report:
                    if errors:
                        report += f"\n\n⚠️ 部分环节有警告: {'; '.join(errors)}"
                    return report
            except Exception as e:
                errors.append(f"报告生成: {e}")
                print(f"[institutional_holdings] format_institutional_report({code}) raised: {e}", file=sys.stderr)

        # ── 2. DB 无数据 (或仅有旧季度), 实时拉取 ──
        holdings = None
        holders = None

        try:
            holdings = get_fund_holdings(code)
        except Exception as e:
            errors.append(f"基金持仓: {e}")
            print(f"[institutional_holdings] get_fund_holdings({code}) raised: {e}", file=sys.stderr)

        try:
            holders = get_top_holders(code)
        except Exception as e:
            errors.append(f"十大股东: {e}")
            print(f"[institutional_holdings] get_top_holders({code}) raised: {e}", file=sys.stderr)

        # ── 3. 尝试入库 (失败不影响报告输出) ──
        try:
            conn = sf._get_db()
            try:
                db.init_schema(conn)
                if holdings and holdings.get("items"):
                    store_fund_holdings(conn, code, holdings)
                if holders and holders.get("items"):
                    store_top_holders(conn, code, holders)
            finally:
                conn.close()
        except Exception as e:
            errors.append(f"入库: {e}")
            print(f"[institutional_holdings] store failed for {code}: {e}", file=sys.stderr)

        # ── 4. 入库后再尝试从 DB 生成完整报告 ──
        try:
            report = format_institutional_report(code)
            if "暂无" not in report:
                if errors:
                    report += f"\n\n⚠️ 部分环节有警告: {'; '.join(errors)}"
                return report
        except Exception as e:
            errors.append(f"报告生成(2): {e}")
            print(f"[institutional_holdings] format_institutional_report({code}) 2nd raised: {e}", file=sys.stderr)

        # ── 4.5 DB 完整报告没成型, 但 DB 里其实有历史数据, 退回用 DB 裸数据拼 ──
        #    (比如当季还没披露, 但上一季度有数据)
        if db_fund_cnt > 0 or db_holder_cnt > 0:
            try:
                lines = [f"机构持仓报告 (历史数据, 当季未披露): {code}", "=" * 55, ""]
                # Open a fresh connection — _db_conn was already closed in the count query above
                _db_conn2 = sf._get_db()
                try:
                    try:
                        name_row = db.fetchone(_db_conn2,
                            "SELECT name FROM stocks WHERE code=?", (code,))
                        if name_row:
                            lines[0] = f"机构持仓报告 (历史数据, 当季未披露): {code} {name_row[0]}"
                    except Exception:
                        pass

                    if db_holder_cnt > 0:
                        hrows = db.fetchall(_db_conn2,
                            "SELECT rank, holder_name, holder_type, hold_pct, change_type, report_date "
                            "FROM stock_top_holders WHERE code = ? "
                            "ORDER BY report_date DESC, rank ASC LIMIT 10", (code,))
                        if hrows:
                            rdate = hrows[0][5] if len(hrows[0]) > 5 else "?"
                            lines.append(f"十大流通股东 (报告期 {rdate}):")
                            type_names = {
                                "fund": "基金", "social_security": "社保", "qfii": "QFII",
                                "insurance": "险资", "broker": "券商", "private": "私募",
                                "connect": "港股通", "individual": "个人", "unknown": "-",
                            }
                            for rnk, hname, htype, pct, ct, _rd in hrows:
                                pct_str = f"{pct:.2f}%" if pct else "-"
                                type_str = type_names.get(htype or "", htype or "-")
                                lines.append(
                                    f"  {rnk:<4} {(hname or '')[:22]:<24} {type_str:<8} "
                                    f"{pct_str:>6} {(ct or '-'):>8}"
                                )
                            lines.append("")

                    if db_fund_cnt > 0:
                        frows = db.fetchall(_db_conn2,
                            "SELECT fund_name, nav_pct, hold_mv, report_date "
                            "FROM fund_holdings WHERE code = ? "
                            "ORDER BY report_date DESC, hold_mv DESC NULLS LAST LIMIT 10",
                            (code,))
                        if frows:
                            rdate = frows[0][3] if len(frows[0]) > 3 else "?"
                            lines.append(f"前10大持仓基金 (共{db_fund_cnt}只历史记录, 最新报告期 {rdate}):")
                            for fname, nav_pct, hold_mv, _rd in frows:
                                nav_str = f"{nav_pct:.2f}%" if nav_pct else "-"
                                mv_str = f"{hold_mv/1e8:.2f}亿" if hold_mv else "-"
                                lines.append(f"  {(fname or '')[:26]:<28} {nav_str:>6} {mv_str:>12}")
                            lines.append("")
                finally:
                    _db_conn2.close()

                if errors:
                    lines.extend(["", f"⚠️ 部分环节有警告: {'; '.join(errors)}"])
                return "\n".join(lines)
            except Exception as e:
                errors.append(f"DB 裸报告: {e}")
                print(f"[institutional_holdings] raw DB report failed for {code}: {e}", file=sys.stderr)

        # ── 5. DB 报告失败, 直接用内存数据拼文本 ──
        lines = [f"机构持仓报告: {code}", "=" * 55, ""]

        if holders and holders.get("items"):
            lines.append(f"十大流通股东 (报告期 {holders.get('report_date', '?')}):")
            lines.append(f"  {'排名':<4} {'股东名称':<24} {'类型':<8} {'占比':>6} {'变动':>8}")
            lines.append("-" * 55)
            type_names = {
                "fund": "基金", "social_security": "社保", "qfii": "QFII",
                "insurance": "险资", "broker": "券商", "private": "私募",
                "connect": "港股通", "individual": "个人", "unknown": "-",
            }
            for it in holders["items"]:
                pct_str = f"{it['hold_pct']:.2f}%" if it.get("hold_pct") else "-"
                type_str = type_names.get(it.get("holder_type", ""), "-")
                lines.append(
                    f"  {it.get('rank',''):<4} {it.get('holder_name','')[:22]:<24} "
                    f"{type_str:<8} {pct_str:>6} {it.get('change_type','-'):>8}"
                )
            lines.append("")

        if holdings and holdings.get("items"):
            lines.append(f"基金持仓 (共{holdings.get('total_count', 0)}只, "
                         f"报告期 {holdings.get('report_date', '?')}):")
            for it in holdings["items"][:10]:
                mv_str = f"{it['hold_mv']/1e8:.2f}亿" if it.get("hold_mv") else "-"
                nav_str = f"{it['nav_pct']:.2f}%" if it.get("nav_pct") else "-"
                lines.append(f"  {it.get('fund_name','')[:26]:<28} {nav_str:>6} {mv_str:>12}")
            lines.append("")

        if not holders and not holdings:
            lines.append("未能获取到机构持仓数据。")

        if errors:
            lines.extend(["", f"⚠️ {'; '.join(errors)}"])

        return "\n".join(lines)

    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}"


# =====================================================
# Tool: Realtime Institutional Signal (近实时机构动向)
# =====================================================

@mcp.tool()
def institutional_realtime(stock_code: str) -> str:
    """
    Query near-realtime institutional activity for an A-share stock.

    Unlike institutional_holdings (quarterly, 15-90 day lag), this tool gives
    a T+1 view based on daily signals:
      - Main-force money flow (stock_moneyflow, daily, 40 points)
      - Dragon-tiger list institutional seats (stock_lhb_stat, daily, 40 points)
      - Block trade activity (stock_dzjy_stat, daily, 20 points)

    Returns a 0-100 score plus breakdown. Combine with institutional_holdings
    to see both the quarterly positioning AND recent activity.

    Use this tool when asked about: 最近机构买入, 近期资金流向, 龙虎榜, 主力资金,
    institutional money flow, recent block trades, smart money activity.

    Args:
        stock_code: Stock code (e.g. "600498") or name (e.g. "烽火通信")
    """
    try:
        resolved = _resolve_stock_unique(stock_code)
        rows = resolved.get("rows", [])
        if not rows:
            return (
                f"ERROR: No stock matched '{stock_code}' in local stocks table."
            )
        if len(rows) != 1 or not resolved.get("code"):
            lines = [
                "ERROR: Stock query is ambiguous.",
                f"Query: {stock_code}",
                "Candidates:",
            ]
            for c, n, s in rows[:10]:
                tag = "停牌" if s else "正常"
                lines.append(f"- {c} {n} ({tag})")
            return "\n".join(lines)
        code = resolved["code"]

        from institutional import (
            format_realtime_report, refresh_lhb_stat, refresh_dzjy_stat,
        )

        # 快照表若为空, 先抓一次(各覆盖全市场, 很快)
        try:
            conn = sf._get_db()
            try:
                db.init_schema(conn)
                lhb_row = db.fetchone(conn,
                    "SELECT COUNT(*) FROM stock_lhb_stat WHERE period = '近一月'")
                if not lhb_row or (lhb_row[0] or 0) == 0:
                    refresh_lhb_stat(conn=conn, period="近一月")
                dzjy_row = db.fetchone(conn,
                    "SELECT COUNT(*) FROM stock_dzjy_stat WHERE period = '近一月'")
                if not dzjy_row or (dzjy_row[0] or 0) == 0:
                    refresh_dzjy_stat(conn=conn, period="近一月")
            finally:
                conn.close()
        except Exception:
            pass

        return format_realtime_report(code)
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}"


# =====================================================
# Start
# =====================================================

def main():
    """Entry point for uvx / pyproject.toml console_scripts."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
