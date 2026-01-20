"""
Base64 Encoder for Matplotlib Figures and PIL Images
Converts visualization outputs to base64 PNG format
"""

import base64
from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def encode_figure_to_base64(fig: Union[Figure, Image.Image], format: str = 'png', dpi: int = 150) -> str:
    """
    Convert matplotlib figure or PIL image to base64 encoded PNG

    Args:
        fig: Matplotlib Figure object or PIL Image
        format: Image format (default: 'png')
        dpi: DPI for saving (only for matplotlib figures)

    Returns:
        Base64 encoded string with data URI prefix
    """
    buf = BytesIO()

    if isinstance(fig, Figure):
        # Matplotlib figure
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    elif isinstance(fig, Image.Image):
        # PIL Image
        fig.save(buf, format=format.upper())
    else:
        raise TypeError(f"Unsupported type: {type(fig)}. Expected Figure or Image.Image")

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Return with data URI prefix for direct embedding
    return f"data:image/{format};base64,{img_base64}"


def save_base64_to_file(base64_str: str, output_path: str) -> None:
    """
    Save base64 encoded image to file

    Args:
        base64_str: Base64 encoded string (with or without data URI prefix)
        output_path: Output file path
    """
    # Remove data URI prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',', 1)[1]

    # Decode and save
    img_data = base64.b64decode(base64_str)
    with open(output_path, 'wb') as f:
        f.write(img_data)
