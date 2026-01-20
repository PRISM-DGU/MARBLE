"""
MARBLE Visualization Style Configuration
All plots are rendered in English with consistent branding
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# MARBLE Brand Colors
MARBLE_COLORS = {
    'primary': '#2E86AB',      # Drug Response Blue
    'secondary': '#A23B72',    # Analysis Purple
    'accent': '#F18F01',       # Warning/Highlight Orange
    'success': '#06A77D',      # Success Green
    'error': '#D00000',        # Error Red

    # Model-specific colors (for 6 supported models)
    'models': {
        'deeptta': '#264653',       # Dark Cyan
        'deepdr': '#F4A261',        # Sandy Brown
        'stagate': '#2A9D8F',       # Persian Green
        'deepst': '#E76F51',        # Burnt Sienna
        'dlm-dti': '#E9C46A',       # Saffron
        'hyperattentiondti': '#9B5DE5',  # Purple
    },

    # Gradient for heatmaps
    'heatmap_cmap': 'RdYlBu_r',

    # Qualitative palette for multiple categories
    'qualitative': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
}

# Figure defaults (all English)
FIGURE_CONFIG = {
    'figsize': (10, 6),
    'dpi': 100,
    'facecolor': 'white',
    'edgecolor': 'none',
}

# Text defaults (English only)
TEXT_CONFIG = {
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'legend_fontsize': 10,
}


def setup_plot_style():
    """
    Configure matplotlib for MARBLE visualizations
    All labels, titles, and legends will be in English
    """
    # Use non-interactive backend
    mpl.use('Agg')

    # Set English font (DejaVu Sans supports scientific symbols)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # Disable fallback to other languages
    plt.rcParams['axes.unicode_minus'] = False

    # Figure settings
    plt.rcParams['figure.figsize'] = FIGURE_CONFIG['figsize']
    plt.rcParams['figure.dpi'] = FIGURE_CONFIG['dpi']
    plt.rcParams['figure.facecolor'] = FIGURE_CONFIG['facecolor']
    plt.rcParams['figure.edgecolor'] = FIGURE_CONFIG['edgecolor']

    # Axes settings
    plt.rcParams['axes.labelsize'] = TEXT_CONFIG['label_fontsize']
    plt.rcParams['axes.titlesize'] = TEXT_CONFIG['title_fontsize']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Tick settings
    plt.rcParams['xtick.labelsize'] = TEXT_CONFIG['tick_fontsize']
    plt.rcParams['ytick.labelsize'] = TEXT_CONFIG['tick_fontsize']

    # Legend settings
    plt.rcParams['legend.fontsize'] = TEXT_CONFIG['legend_fontsize']
    plt.rcParams['legend.frameon'] = False

    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5

    # Savefig settings (for high-quality output)
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


def get_model_color(model_name: str) -> str:
    """
    Get consistent color for a specific model

    Args:
        model_name: Model identifier (deeptta, deepdr, stagate, deepst, dlm-dti, hyperattentiondti)

    Returns:
        Hex color code
    """
    model_name_lower = model_name.lower()
    return MARBLE_COLORS['models'].get(model_name_lower, MARBLE_COLORS['primary'])


def get_qualitative_colors(n: int) -> list:
    """
    Get n distinct colors for categorical data

    Args:
        n: Number of colors needed

    Returns:
        List of hex color codes
    """
    colors = MARBLE_COLORS['qualitative']
    if n <= len(colors):
        return colors[:n]
    # Repeat if more colors needed
    return (colors * ((n // len(colors)) + 1))[:n]
