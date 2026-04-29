"""
通用工具函数 — 供各分析模块共享
================================
提供:
  _extract_float()            → 从文本中用正则提取浮点数
  _extract_float_warn()       → 带提取失败监控的版本
  _has_keyword_unaffirmed()   → 否定语义感知的关键词检测
  _module_header()            → 模块报告标准头部行列表
"""

import re
import logging
from typing import Optional

_log = logging.getLogger(__name__)

# 同一标签连续提取失败计数（≥3次时发出WARNING）
_extraction_fail_counts: dict = {}


def _extract_float(text: str, pattern: str) -> Optional[float]:
    """从文本中用正则提取浮点数。text 为空时直接返回 None。"""
    if not text:
        return None
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return None


def _extract_float_warn(text: str, pattern: str, label: str = "") -> Optional[float]:
    """_extract_float 的带失败监控版本。

    当同一 label 连续 ≥3 次从非空有效文本中提取失败时，发出 WARNING。
    这通常意味着 MCP 工具输出格式已变更，导致信号静默降级为中性值。
    """
    result = _extract_float(text, pattern)

    is_valid_text = (
        text and len(text) > 20
        and not text.startswith("⏸")
        and "失败" not in text[:40]
        and "未提供" not in text[:40]
    )

    if result is None and is_valid_text:
        count = _extraction_fail_counts.get(label, 0) + 1
        _extraction_fail_counts[label] = count
        if count >= 3 and count % 3 == 0:
            _log.warning(
                "信号提取连续失败 %d 次 [%s]，工具输出格式可能已变更 | pattern=%s | text[:60]=%r",
                count, label, pattern, text[:60],
            )
    elif result is not None and label:
        _extraction_fail_counts[label] = 0  # 成功后重置

    return result


# ── 中文否定词表 ────────────────────────────────────────────
# 单字否定（紧贴关键词前一字）
_NEG_SINGLE = frozenset("无未不非")
# 多字否定短语（关键词前8字内出现即判定为否定语境）
_NEG_PHRASES = (
    "没有", "并无", "并不", "并未", "尚未", "暂无",
    "不存在", "未出现", "未发生", "未见", "未发现",
    "不存在任何", "未有任何",
)


def _has_keyword_unaffirmed(text: str, keywords) -> bool:
    """检查 text 中是否存在未被中文否定词修饰的关键词。

    对每个关键词的每处匹配，检查其前 8 个字符是否含否定词。
    只要存在一处未被否定的匹配，即返回 True。

    示例:
      _has_keyword_unaffirmed("公司出现涨停", ["涨停"])       → True
      _has_keyword_unaffirmed("公司并未出现涨停", ["涨停"])   → False
      _has_keyword_unaffirmed("无利空消息", ["利空"])         → False
      _has_keyword_unaffirmed("存在利空风险", ["利空"])       → True
    """
    if not text:
        return False
    for kw in keywords:
        idx = text.find(kw)
        while idx != -1:
            prefix = text[max(0, idx - 8): idx]
            # 多字短语检查
            if any(neg in prefix for neg in _NEG_PHRASES):
                idx = text.find(kw, idx + 1)
                continue
            # 单字紧邻检查
            if prefix and prefix[-1] in _NEG_SINGLE:
                idx = text.find(kw, idx + 1)
                continue
            return True  # 找到一处未被否定的匹配
        # 当前 kw 所有匹配均被否定，继续检查下一个 kw
    return False


def _module_header(title: str, stock_name: str = "", stock_code: str = "") -> list:
    """返回模块报告的标准头部行，供各 format_*() 函数解包到列表开头。"""
    suffix = f" — {stock_name}（{stock_code}）" if stock_name else ""
    return ["", "=" * 60, f"{title}{suffix}", "=" * 60, ""]
