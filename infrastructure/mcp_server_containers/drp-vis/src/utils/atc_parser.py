"""
ATC Code Hierarchical Parser
Parse ATC (Anatomical Therapeutic Chemical) codes into hierarchical levels

ATC Code Structure:
- Level 1: Anatomical main group (1 letter)          e.g., "L"
- Level 2: Therapeutic subgroup (2 digits)           e.g., "L01"
- Level 3: Pharmacological subgroup (1 letter)       e.g., "L01X"
- Level 4: Chemical subgroup (1 letter)              e.g., "L01XE"
- Level 5: Chemical substance (2 digits)             e.g., "L01XE01"

Example: L01XE01 (Imatinib)
- L: Antineoplastic and immunomodulating agents
- L01: Antineoplastic agents
- L01X: Other antineoplastic agents
- L01XE: Protein kinase inhibitors
- L01XE01: Imatinib
"""

import pandas as pd
from typing import Optional


# ATC Level descriptions
ATC_LEVEL_NAMES = {
    1: "Anatomical Main Group",
    2: "Therapeutic Subgroup",
    3: "Pharmacological Subgroup",
    4: "Chemical Subgroup",
    5: "Chemical Substance"
}


def parse_atc_level(atc_code: str, level: int = 5) -> Optional[str]:
    """
    Extract ATC code at specified hierarchical level

    Args:
        atc_code: Full ATC code (e.g., "L01XE01")
        level: Hierarchical level (1-5)
               1 = Anatomical (1 char)
               2 = Therapeutic (3 chars)
               3 = Pharmacological (4 chars)
               4 = Chemical (5 chars)
               5 = Substance (7 chars, full code)

    Returns:
        ATC code at specified level, or None if invalid

    Examples:
        >>> parse_atc_level("L01XE01", 1)
        "L"
        >>> parse_atc_level("L01XE01", 2)
        "L01"
        >>> parse_atc_level("L01XE01", 4)
        "L01XE"
        >>> parse_atc_level("L01XE01", 5)
        "L01XE01"
    """
    if not isinstance(atc_code, str) or not atc_code:
        return None

    # Define character positions for each level
    level_length = {
        1: 1,   # L
        2: 3,   # L01
        3: 4,   # L01X
        4: 5,   # L01XE
        5: 7    # L01XE01 (or variable length for full code)
    }

    if level not in level_length:
        return None

    # For level 5, return full code
    if level == 5:
        return atc_code

    # For other levels, slice to appropriate length
    target_length = level_length[level]
    if len(atc_code) < target_length:
        return None

    return atc_code[:target_length]


def get_atc_level_name(level: int) -> str:
    """
    Get descriptive name for ATC level

    Args:
        level: ATC hierarchical level (1-5)

    Returns:
        Level name string

    Examples:
        >>> get_atc_level_name(1)
        "Anatomical Main Group"
        >>> get_atc_level_name(4)
        "Chemical Subgroup"
    """
    return ATC_LEVEL_NAMES.get(level, f"Level {level}")


def add_atc_levels_to_dataframe(df: pd.DataFrame,
                                 atc_column: str = 'ATC_CODE',
                                 levels: list = [1, 2, 3, 4, 5]) -> pd.DataFrame:
    """
    Add ATC level columns to DataFrame

    Args:
        df: DataFrame with ATC codes
        atc_column: Column name containing ATC codes
        levels: List of levels to add (default: all levels 1-5)

    Returns:
        DataFrame with additional columns: ATC_LEVEL_1, ATC_LEVEL_2, etc.

    Example:
        >>> df = pd.DataFrame({'ATC_CODE': ['L01XE01', 'L01BC02']})
        >>> df = add_atc_levels_to_dataframe(df)
        >>> df.columns
        ['ATC_CODE', 'ATC_LEVEL_1', 'ATC_LEVEL_2', 'ATC_LEVEL_3', 'ATC_LEVEL_4', 'ATC_LEVEL_5']
    """
    df = df.copy()

    if atc_column not in df.columns:
        raise ValueError(f"Column '{atc_column}' not found in DataFrame")

    for level in levels:
        col_name = f'ATC_LEVEL_{level}'
        df[col_name] = df[atc_column].apply(lambda x: parse_atc_level(x, level))

    return df


def group_by_atc_level(df: pd.DataFrame,
                       atc_column: str = 'ATC_CODE',
                       level: int = 4,
                       agg_func: dict = None) -> pd.DataFrame:
    """
    Group DataFrame by ATC level and aggregate

    Args:
        df: DataFrame with ATC codes
        atc_column: Column name containing ATC codes
        level: ATC level to group by (1-5)
        agg_func: Aggregation functions (default: mean for numeric columns)

    Returns:
        Grouped and aggregated DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'ATC_CODE': ['L01XE01', 'L01XE03', 'L01BC02'],
        ...     'IC50': [2.3, 2.5, 1.8]
        ... })
        >>> grouped = group_by_atc_level(df, level=4, agg_func={'IC50': 'mean'})
    """
    # Add level column if not exists
    level_col = f'ATC_LEVEL_{level}'
    if level_col not in df.columns:
        df = add_atc_levels_to_dataframe(df, atc_column, levels=[level])

    # Default aggregation: mean for numeric columns
    if agg_func is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        agg_func = {col: 'mean' for col in numeric_cols}

    # Group and aggregate
    grouped = df.groupby(level_col).agg(agg_func).reset_index()

    return grouped


def get_atc_hierarchy_dict(atc_codes: list) -> dict:
    """
    Build hierarchical dictionary from list of ATC codes

    Args:
        atc_codes: List of ATC codes

    Returns:
        Nested dictionary representing hierarchy

    Example:
        >>> codes = ['L01XE01', 'L01XE03', 'L01BC02']
        >>> hierarchy = get_atc_hierarchy_dict(codes)
        >>> hierarchy['L']['L01']['L01X']['L01XE']
        ['L01XE01', 'L01XE03']
    """
    hierarchy = {}

    for code in atc_codes:
        if not isinstance(code, str) or not code:
            continue

        # Parse all levels
        l1 = parse_atc_level(code, 1)
        l2 = parse_atc_level(code, 2)
        l3 = parse_atc_level(code, 3)
        l4 = parse_atc_level(code, 4)
        l5 = code

        # Build nested structure
        if l1 not in hierarchy:
            hierarchy[l1] = {}
        if l2 not in hierarchy[l1]:
            hierarchy[l1][l2] = {}
        if l3 not in hierarchy[l1][l2]:
            hierarchy[l1][l2][l3] = {}
        if l4 not in hierarchy[l1][l2][l3]:
            hierarchy[l1][l2][l3][l4] = []

        hierarchy[l1][l2][l3][l4].append(l5)

    return hierarchy
