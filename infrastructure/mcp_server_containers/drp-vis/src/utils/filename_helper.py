"""
Filename Helper Utility
Generates timestamped filenames for saved visualizations
"""

from datetime import datetime


def get_timestamp() -> str:
    """
    Get current timestamp in MMDD-HHMM format

    Returns:
        Timestamp string (e.g., "1029-1110")
    """
    now = datetime.now()
    return now.strftime("%m%d-%H%M")


def get_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """
    Generate filename (without timestamp)

    Args:
        base_name: Base filename without extension (e.g., "predict_vs_actual")
        extension: File extension without dot (default: "png")

    Returns:
        Filename (e.g., "predict_vs_actual.png")
    """
    return f"{base_name}.{extension}"
