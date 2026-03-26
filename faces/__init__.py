"""
A股分析五面体系
===============
按分析维度拆分，每个"面"封装自己的数据获取 + 分析 + 评分逻辑。

短线调用: 技术面 + 资金面 + 催化剂面 + 风控面
长线调用: 基本面 + 催化剂面(部分) + 风控面
共同依赖: 环境层 (market_news + northbound_flow → H/M/L)

每个面模块对外暴露统一接口:
  - analyze(stock_code, **kwargs) → FaceResult
  - score(signals) → DimensionScore(s)
"""

from .face_technical import TechnicalFace
from .face_capital import CapitalFace
from .face_catalyst import CatalystFace
from .face_fundamental import FundamentalFace
from .face_risk import RiskFace
from .scorecard import Scorecard, compute_combined_decision, format_combined_decision

__all__ = [
    "TechnicalFace",
    "CapitalFace",
    "CatalystFace",
    "FundamentalFace",
    "RiskFace",
    "Scorecard",
    "compute_combined_decision",
    "format_combined_decision",
]
