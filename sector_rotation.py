"""
Sector Rotation Model for Chinese A-shares.

Implements sector classification, metrics computation, momentum ranking,
and rotation signal generation based on historical price and volume data.
Uses only numpy and pandas for computations.
"""

import db
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Standard sector mapping for A-shares
SECTORS = {
    "金融": "finance",
    "房地产": "real_estate",
    "消费": "consumer",
    "医药": "healthcare",
    "科技": "technology",
    "制造": "manufacturing",
    "能源": "energy",
    "材料": "materials",
    "公用事业": "utilities",
    "交通运输": "transportation",
    "军工": "defense",
    "文化传媒": "media",
    "建筑": "construction",
    "农业": "agriculture",
    "其他": "other",
}

# Reverse mapping: English to Chinese sector names
SECTOR_CN = {v: k for k, v in SECTORS.items()}

# Sector classification keywords (Chinese)
SECTOR_KEYWORDS = {
    "金融": ["银行", "证券", "保险", "信托", "金融", "期货"],
    "房地产": ["地产", "置业", "房产", "置地"],
    "医药": ["医药", "药业", "生物", "医疗", "制药", "健康", "基因"],
    "军工": ["军工", "国防", "航天", "兵器", "导弹"],
    # 文化传媒 must appear BEFORE 科技: "游戏" stocks with "科技" in name map to media, not tech
    "文化传媒": ["传媒", "游戏", "动画", "电影", "音乐", "影视", "出版", "广告", "媒体", "文化传媒"],
    "科技": ["科技", "软件", "信息", "电子", "半导体", "芯片", "智能", "数据",
             "网络", "通信", "计算机", "互联", "云计算", "大数据", "人工智能",
             "服务器", "存储", "5G"],
    "消费": ["食品", "饮料", "酒", "乳业", "家电", "服饰", "零售", "百货", "超市",
             "旅游", "酒店", "餐饮", "教育"],
    "制造": ["机械", "装备", "汽车", "电气", "自动化", "仪器", "航空", "船舶",
             "设备", "工业", "精密"],
    "能源": ["能源", "电力", "煤炭", "石油", "天然气", "石化", "新能源", "光伏",
             "风电", "核电", "储能", "充电桩", "锂电", "燃料电池"],
    "材料": ["钢铁", "有色", "化工", "建材", "水泥", "玻璃", "铝", "铜", "锂"],
    "公用事业": ["水务", "燃气", "环保", "供热", "供水"],
    "交通运输": ["交通", "运输", "物流", "航运", "港口", "铁路", "公路", "快递",
                "机场", "仓储", "货运"],
    "建筑": ["建筑", "建设", "工程", "路桥", "市政", "施工", "基建"],
    "农业": ["农业", "种业", "养殖", "渔业", "牧业", "农产品", "畜牧"],
}


def classify_sector(code: str, name: str) -> str:
    """
    Classify stock into sector based on name keywords.

    Args:
        code: Stock code (6-digit string)
        name: Stock name (Chinese)

    Returns:
        Sector key from SECTORS dict (English name)
    """
    # Strip whitespace
    name = name.strip()

    # Match keywords
    for sector_cn, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name:
                return SECTORS[sector_cn]

    # Default to other
    return SECTORS["其他"]


def _load_stocks_with_sectors() -> pd.DataFrame:
    """
    Load all stocks from DB with sector classification.

    Returns:
        DataFrame with columns: code, name, sector
    """
    conn = db.get_conn()
    try:
        df = db.read_sql(
            "SELECT code, name, suspended FROM stocks WHERE suspended = 0",
            conn
        )
    finally:
        conn.close()

    df['sector'] = df.apply(
        lambda row: classify_sector(row['code'], row['name']),
        axis=1
    )

    return df[['code', 'name', 'sector']]


def _load_price_history(days: int = 60) -> pd.DataFrame:
    """
    Load price history for all stocks.

    Args:
        days: Number of recent trading days to load

    Returns:
        DataFrame with columns: code, date, close, volume, pct_chg
    """
    conn = db.get_conn()
    try:
        # Get the most recent trading date first
        max_date_query = "SELECT MAX(date) as max_date FROM stock_history"
        max_date = db.read_sql(max_date_query, conn)['max_date'].iloc[0]

        # Load history from that date going back
        query = f"""
            SELECT code, date, close, volume, pct_chg
            FROM stock_history
            WHERE date <= '{max_date}'
            ORDER BY code, date DESC
        """
        df = db.read_sql(query, conn)
    finally:
        conn.close()

    # Get unique codes and keep last N days for each
    df = df.sort_values(['code', 'date'], ascending=[True, False])
    df = df.groupby('code').head(days).reset_index(drop=True)

    # Sort for further processing
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    return df


def _compute_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def get_sector_map() -> Dict[str, str]:
    """
    Return {code: sector} mapping for all stocks in DB.

    Returns:
        Dictionary mapping stock code to sector (English name)
    """
    df = _load_stocks_with_sectors()
    return dict(zip(df['code'], df['sector']))


def compute_sector_metrics(days: int = 60) -> pd.DataFrame:
    """
    Compute metrics for each sector.

    Args:
        days: Number of recent trading days to analyze

    Returns:
        DataFrame indexed by sector with columns:
        - return_5d, return_20d, return_60d: Returns over period
        - momentum_score: Composite momentum score (0-100)
        - volume_change: Average volume change %
        - breadth: % of stocks above MA20
        - relative_strength: Sector return vs market return
        - volatility: Annualized volatility
        - stock_count: Number of stocks in sector
    """
    stocks_df = _load_stocks_with_sectors()
    history_df = _load_price_history(days=days)

    # Merge stock info with history
    df = history_df.merge(stocks_df[['code', 'sector']], on='code', how='left')
    df = df.dropna(subset=['sector'])

    # Ensure proper data types
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
    df = df.dropna(subset=['close', 'pct_chg'])

    results = []

    for sector in df['sector'].unique():
        sector_data = df[df['sector'] == sector].copy()

        if sector_data.empty or len(sector_data['code'].unique()) < 5:
            continue

        # Compute metrics per stock first
        stock_metrics = []
        for code, group in sector_data.groupby('code'):
            group = group.sort_values('date').reset_index(drop=True)

            if len(group) < 5:
                continue

            # Returns over different periods
            close_prices = group['close'].values
            pct_changes = group['pct_chg'].fillna(0).values

            # Compute period returns
            ret_5d = (1 + pct_changes[-5:].sum() / 100) - 1 if len(group) >= 5 else 0
            ret_20d = (1 + pct_changes[-20:].sum() / 100) - 1 if len(group) >= 20 else 0
            ret_60d = (1 + pct_changes[-60:].sum() / 100) - 1 if len(group) >= 60 else 0

            # Volume change (last 5d vs previous 5d)
            volumes = group['volume'].values
            if len(volumes) >= 10:
                vol_recent = volumes[-5:].mean()
                vol_prev = volumes[-10:-5].mean()
                vol_change = (vol_recent / vol_prev - 1) * 100 if vol_prev > 0 else 0
            else:
                vol_change = 0

            # Breadth: % above MA20
            ma20 = _compute_moving_average(group['close'], 20).values
            above_ma20 = (close_prices >= ma20).sum() / len(close_prices)

            stock_metrics.append({
                'code': code,
                'return_5d': ret_5d,
                'return_20d': ret_20d,
                'return_60d': ret_60d,
                'volume_change': vol_change,
                'breadth': above_ma20,
                'volatility': group['pct_chg'].std() * np.sqrt(252) if len(group) > 1 else 0,
            })

        if not stock_metrics:
            continue

        stock_df = pd.DataFrame(stock_metrics)

        # Aggregate to sector level (equal-weighted)
        sector_return_5d = stock_df['return_5d'].mean()
        sector_return_20d = stock_df['return_20d'].mean()
        sector_return_60d = stock_df['return_60d'].mean()
        sector_volume_change = stock_df['volume_change'].mean()
        sector_breadth = stock_df['breadth'].mean()
        sector_volatility = stock_df['volatility'].mean()

        # Momentum score (0-100)
        # Components: recent returns (weight 0.5), breadth (weight 0.3), volume change (weight 0.2)
        normalized_return = max(0, min(100, (sector_return_20d * 1000 + 50)))
        normalized_breadth = sector_breadth * 100
        normalized_volume = max(0, min(100, sector_volume_change + 50))

        momentum_score = (
            normalized_return * 0.5 +
            normalized_breadth * 0.3 +
            normalized_volume * 0.2
        )

        # Relative strength vs market (global market return)
        market_return = df.groupby('code')['pct_chg'].apply(
            lambda x: (1 + x.sum() / 100) - 1
        ).mean()
        relative_strength = sector_return_20d - market_return if market_return != 0 else 0

        results.append({
            'sector': sector,
            'return_5d': sector_return_5d,
            'return_20d': sector_return_20d,
            'return_60d': sector_return_60d,
            'momentum_score': momentum_score,
            'volume_change': sector_volume_change,
            'breadth': sector_breadth,
            'relative_strength': relative_strength,
            'volatility': sector_volatility,
            'stock_count': len(stock_df),
        })

    result_df = pd.DataFrame(results).set_index('sector')
    return result_df


def sector_momentum_ranking(days: int = 20) -> List[Dict]:
    """
    Rank sectors by momentum.

    Args:
        days: Number of days for momentum calculation

    Returns:
        Sorted list of dicts with keys:
        - sector: Sector name (English)
        - score: Momentum score (0-100)
        - return_20d: 20-day return
        - breadth: % stocks above MA20
        - signal: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    """
    metrics = compute_sector_metrics(days=days)

    if metrics.empty:
        return []

    rankings = []

    # Normalize momentum scores for signal generation
    max_score = metrics['momentum_score'].max()
    min_score = metrics['momentum_score'].min()
    score_range = max_score - min_score if max_score > min_score else 1

    for sector in metrics.index:
        row = metrics.loc[sector]
        score = row['momentum_score']

        # Normalize to 0-100
        norm_score = ((score - min_score) / score_range * 100) if score_range > 0 else 50

        # Generate signal based on normalized score and breadth
        if norm_score >= 75 and row['breadth'] >= 0.6:
            signal = 'strong_buy'
        elif norm_score >= 60 and row['breadth'] >= 0.5:
            signal = 'buy'
        elif norm_score >= 40 and norm_score < 60:
            signal = 'hold'
        elif norm_score >= 25 and row['breadth'] < 0.5:
            signal = 'sell'
        else:
            signal = 'strong_sell'

        rankings.append({
            'sector': sector,
            'score': float(norm_score),
            'return_20d': float(row['return_20d']),
            'breadth': float(row['breadth']),
            'volatility': float(row['volatility']),
            'signal': signal,
        })

    # Sort by score descending
    rankings.sort(key=lambda x: x['score'], reverse=True)

    return rankings


def sector_rotation_signal() -> Dict:
    """
    Generate sector rotation recommendation.

    Based on:
    1. Relative momentum (which sectors are leading)
    2. Mean reversion (which oversold sectors may bounce)
    3. Breadth divergence (sector with improving breadth)

    Returns:
        Dict with keys:
        - overweight: List of {sector, reason, score}
        - underweight: List of {sector, reason, score}
        - neutral: List of {sector}
        - rotation_phase: 'early_cycle', 'mid_cycle', 'late_cycle', 'defensive'
        - summary: Summary string
    """
    metrics = compute_sector_metrics(days=60)

    if metrics.empty:
        return {
            'overweight': [],
            'underweight': [],
            'neutral': [],
            'rotation_phase': 'unknown',
            'summary': 'Insufficient data for rotation analysis',
        }

    # Identify leaders and laggards
    metrics['momentum_rank'] = metrics['momentum_score'].rank(ascending=False)
    metrics['return_rank'] = metrics['return_60d'].rank(ascending=False)
    metrics['breadth_rank'] = metrics['breadth'].rank(ascending=False)

    # Composite rank
    metrics['composite_rank'] = (
        metrics['momentum_rank'] * 0.5 +
        metrics['return_rank'] * 0.3 +
        metrics['breadth_rank'] * 0.2
    )

    # Sort by composite rank
    metrics = metrics.sort_values('composite_rank')

    num_sectors = len(metrics)
    overweight_count = max(2, num_sectors // 3)
    underweight_count = max(2, num_sectors // 3)

    overweight_list = []
    underweight_list = []
    neutral_list = []

    # Overweight: top sectors
    for sector in metrics.head(overweight_count).index:
        row = metrics.loc[sector]

        reasons = []
        score = 0

        if row['momentum_score'] > metrics['momentum_score'].median():
            reasons.append("Strong momentum")
            score += 30
        if row['return_20d'] > 0:
            reasons.append("Positive 20d return")
            score += 25
        if row['breadth'] > 0.5:
            reasons.append("Good breadth")
            score += 25
        if row['relative_strength'] > 0:
            reasons.append("Outperforming market")
            score += 20

        overweight_list.append({
            'sector': sector,
            'reason': '; '.join(reasons) if reasons else 'Technical setup favorable',
            'score': float(score),
        })

    # Underweight: bottom sectors
    for sector in metrics.tail(underweight_count).index:
        row = metrics.loc[sector]

        reasons = []
        score = 0

        if row['momentum_score'] < metrics['momentum_score'].median():
            reasons.append("Weak momentum")
            score += 30
        if row['return_20d'] < 0:
            reasons.append("Negative 20d return")
            score += 25
        if row['breadth'] < 0.4:
            reasons.append("Poor breadth")
            score += 25
        if row['relative_strength'] < 0:
            reasons.append("Underperforming market")
            score += 20

        underweight_list.append({
            'sector': sector,
            'reason': '; '.join(reasons) if reasons else 'Technical setup weak',
            'score': float(score),
        })

    # Neutral: middle sectors
    neutral_sectors = metrics.iloc[overweight_count:-underweight_count].index.tolist()
    neutral_list = [{'sector': s} for s in neutral_sectors]

    # Detect rotation phase based on sector leadership
    overweight_sectors_set = set([x['sector'] for x in overweight_list])

    # Define sector groups — materials removed from late_cycle to avoid dual membership
    early_cycle_sectors = {'finance', 'materials', 'energy'}
    mid_cycle_sectors = {'technology', 'consumer'}
    late_cycle_sectors = {'energy'}  # materials belongs to early_cycle only
    defensive_sectors = {'utilities', 'healthcare'}

    early_cycle_score = len(overweight_sectors_set & early_cycle_sectors)
    mid_cycle_score = len(overweight_sectors_set & mid_cycle_sectors)
    late_cycle_score = len(overweight_sectors_set & late_cycle_sectors)
    defensive_score = len(overweight_sectors_set & defensive_sectors)

    rotation_phase = 'unknown'
    if early_cycle_score >= 1:
        rotation_phase = 'early_cycle'
    elif mid_cycle_score >= 1:
        rotation_phase = 'mid_cycle'
    elif late_cycle_score >= 1:
        rotation_phase = 'late_cycle'
    elif defensive_score >= 1:
        rotation_phase = 'defensive'

    # Generate summary
    top_sector = overweight_list[0]['sector'] if overweight_list else 'None'
    bottom_sector = underweight_list[0]['sector'] if underweight_list else 'None'

    summary = (
        f"Rotation phase: {rotation_phase}. "
        f"Top sector: {top_sector}. "
        f"Weakest sector: {bottom_sector}. "
        f"Overweight {len(overweight_list)} sector(s), underweight {len(underweight_list)} sector(s)."
    )

    return {
        'overweight': overweight_list,
        'underweight': underweight_list,
        'neutral': neutral_list,
        'rotation_phase': rotation_phase,
        'summary': summary,
    }


def sector_correlation_matrix(days: int = 60) -> pd.DataFrame:
    """
    Compute correlation between sector returns.

    Args:
        days: Number of recent trading days to analyze

    Returns:
        Correlation matrix (DataFrame) indexed and columned by sector
    """
    stocks_df = _load_stocks_with_sectors()
    history_df = _load_price_history(days=days)

    # Merge stock info with history
    df = history_df.merge(stocks_df[['code', 'sector']], on='code', how='left')
    df = df.dropna(subset=['sector'])

    # Ensure proper data types
    df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
    df = df.dropna(subset=['pct_chg'])

    # Compute sector-level returns (daily equal-weighted)
    df['date'] = pd.to_datetime(df['date'])
    sector_returns = df.groupby(['date', 'sector'])['pct_chg'].mean().unstack(fill_value=0)

    # Compute correlation
    correlation = sector_returns.corr()

    return correlation


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("SECTOR ROTATION MODEL - SECTOR METRICS")
    print("=" * 80)

    metrics = compute_sector_metrics(days=60)
    if not metrics.empty:
        print(metrics[['return_20d', 'momentum_score', 'breadth', 'volatility']].round(3))
        print()

    print("=" * 80)
    print("SECTOR MOMENTUM RANKING")
    print("=" * 80)

    rankings = sector_momentum_ranking(days=20)
    for rank in rankings:
        print(
            f"{rank['sector']:15} | Score: {rank['score']:6.1f} | "
            f"Return: {rank['return_20d']:7.2%} | Breadth: {rank['breadth']:5.1%} | "
            f"Signal: {rank['signal']}"
        )
    print()

    print("=" * 80)
    print("SECTOR ROTATION SIGNAL")
    print("=" * 80)

    signal = sector_rotation_signal()

    print(f"Rotation Phase: {signal['rotation_phase']}")
    print(f"Summary: {signal['summary']}")
    print()

    if signal['overweight']:
        print("OVERWEIGHT:")
        for item in signal['overweight']:
            print(f"  {item['sector']:15} Score: {item['score']:5.1f} | {item['reason']}")
    print()

    if signal['underweight']:
        print("UNDERWEIGHT:")
        for item in signal['underweight']:
            print(f"  {item['sector']:15} Score: {item['score']:5.1f} | {item['reason']}")
    print()

    print("=" * 80)
    print("SECTOR CORRELATION MATRIX")
    print("=" * 80)

    corr = sector_correlation_matrix(days=60)
    if not corr.empty:
        print(corr.round(3))
    print()
