"""
Cancer Type × ATC Code Heatmap Visualization
Visualize IC50 metrics in heatmap format grouped by cancer type and ATC code
"""

import json
import logging
import os
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plot_style import setup_plot_style, AUTODRP_COLORS
from utils.filename_helper import get_timestamped_filename
from utils.atc_parser import parse_atc_level, add_atc_levels_to_dataframe

logger = logging.getLogger(__name__)


class CancerATCHeatmapVisualizationTool:
    """Visualize Cancer Type × ATC Code heatmap for IC50 metrics"""

    def __init__(self):
        setup_plot_style()

    def apply(self,
              model_csv_path: str,
              model_name: str,
              metric: Literal['mean_ic50', 'median_ic50', 'rmse', 'bias'] = 'mean_ic50',
              atc_level: int = 4,
              value_type: Literal['y_true', 'y_pred', 'both'] = 'y_true',
              output_dir: str = "figures") -> str:
        """
        Generate Cancer Type × ATC Code heatmap

        Args:
            model_csv_path: Path to model CSV file
                           CSV must have: CELL_LINE_NAME, CANCER_TYPE, DRUG_NAME, ATC_CODE, y_true, y_pred
            model_name: Model name for title (e.g., "deepdr")
            metric: Metric to visualize
                   'mean_ic50': Mean IC50 value
                   'median_ic50': Median IC50 value
                   'rmse': Root mean square error (requires y_true and y_pred)
                   'bias': Prediction bias (y_pred - y_true)
            atc_level: ATC hierarchical level (1-5)
                      1 = Anatomical, 2 = Therapeutic, 3 = Pharmacological,
                      4 = Chemical subgroup, 5 = Chemical substance
            value_type: Which value to use for metric calculation
                       'y_true': Use experimental values (GDSC2)
                       'y_pred': Use model predictions
                       'both': Use both (for rmse, bias)
            output_dir: Output directory for PNG (default: "figures")

        Returns:
            JSON string with status, output path, and heatmap statistics
        """
        try:
            # Validate input
            if not os.path.exists(model_csv_path):
                return json.dumps({
                    "status": "error",
                    "message": f"CSV file not found: {model_csv_path}"
                })

            # Load data
            df = pd.read_csv(model_csv_path)
            required_cols = ['CELL_LINE_NAME', 'CANCER_TYPE', 'DRUG_NAME', 'ATC_CODE', 'y_true', 'y_pred']
            if not all(col in df.columns for col in required_cols):
                return json.dumps({
                    "status": "error",
                    "message": f"CSV must have columns: {required_cols}"
                })

            # Add ATC level column
            df = add_atc_levels_to_dataframe(df, atc_column='ATC_CODE', levels=[atc_level])
            atc_col = f'ATC_LEVEL_{atc_level}'

            # Calculate metric for each (cancer_type, atc_code) combination
            heatmap_data = self._calculate_heatmap_data(df, atc_col, metric, value_type)

            if heatmap_data is None or heatmap_data.empty:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to calculate heatmap data"
                })

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))

            # Choose colormap based on metric
            if metric in ['rmse']:
                cmap = 'YlOrRd'  # Yellow-Orange-Red for error metrics
                vmin = 0
                vmax = None
            elif metric == 'bias':
                cmap = 'RdBu_r'  # Red-Blue diverging for bias
                abs_max = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
                vmin = -abs_max
                vmax = abs_max
            else:
                cmap = 'viridis'  # Default colormap
                vmin = None
                vmax = None

            # Plot heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap,
                       cbar_kws={'label': self._get_metric_label(metric)},
                       linewidths=0.5, linecolor='gray',
                       vmin=vmin, vmax=vmax, ax=ax)

            # Labels and title
            ax.set_xlabel('ATC Code', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

            metric_name = self._get_metric_label(metric)
            value_suffix = {
                'y_true': ' (GDSC2 Experimental)',
                'y_pred': f' ({model_name.upper()} Predicted)',
                'both': ''
            }[value_type]

            title = f'{model_name.upper()}: {metric_name} by Cancer Type and ATC Code (Level {atc_level}){value_suffix}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            plt.tight_layout()

            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            base_filename = f"heatmap_{metric}_{model_name}_atc{atc_level}"
            filename = get_timestamped_filename(base_filename, "png")
            output_path = os.path.join(output_dir, filename)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Calculate statistics
            stats = {
                'overall_mean': float(heatmap_data.values.flatten().mean()),
                'overall_std': float(np.nanstd(heatmap_data.values.flatten())),
                'overall_min': float(heatmap_data.min().min()),
                'overall_max': float(heatmap_data.max().max()),
                'n_cancer_types': len(heatmap_data.index),
                'n_atc_codes': len(heatmap_data.columns),
                'cancer_types': heatmap_data.index.tolist(),
                'atc_codes': heatmap_data.columns.tolist()
            }

            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "metric": metric,
                "atc_level": atc_level,
                "value_type": value_type,
                "statistics": stats,
                "metadata": {
                    "model_name": model_name,
                    "output_file": filename
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"Cancer-ATC heatmap visualization error: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })

    def _calculate_heatmap_data(self, df: pd.DataFrame, atc_col: str,
                                metric: str, value_type: str) -> Optional[pd.DataFrame]:
        """Calculate metric values for heatmap"""
        try:
            if metric == 'mean_ic50':
                if value_type == 'y_true':
                    pivot = df.groupby(['CANCER_TYPE', atc_col])['y_true'].mean().unstack(fill_value=np.nan)
                elif value_type == 'y_pred':
                    pivot = df.groupby(['CANCER_TYPE', atc_col])['y_pred'].mean().unstack(fill_value=np.nan)
                else:
                    # Average of both
                    pivot_true = df.groupby(['CANCER_TYPE', atc_col])['y_true'].mean().unstack(fill_value=np.nan)
                    pivot_pred = df.groupby(['CANCER_TYPE', atc_col])['y_pred'].mean().unstack(fill_value=np.nan)
                    pivot = (pivot_true + pivot_pred) / 2

            elif metric == 'median_ic50':
                if value_type == 'y_true':
                    pivot = df.groupby(['CANCER_TYPE', atc_col])['y_true'].median().unstack(fill_value=np.nan)
                elif value_type == 'y_pred':
                    pivot = df.groupby(['CANCER_TYPE', atc_col])['y_pred'].median().unstack(fill_value=np.nan)
                else:
                    pivot_true = df.groupby(['CANCER_TYPE', atc_col])['y_true'].median().unstack(fill_value=np.nan)
                    pivot_pred = df.groupby(['CANCER_TYPE', atc_col])['y_pred'].median().unstack(fill_value=np.nan)
                    pivot = (pivot_true + pivot_pred) / 2

            elif metric == 'rmse':
                def calc_rmse(group):
                    return np.sqrt(np.mean((group['y_true'] - group['y_pred']) ** 2))
                pivot = df.groupby(['CANCER_TYPE', atc_col]).apply(calc_rmse).unstack(fill_value=np.nan)

            elif metric == 'bias':
                def calc_bias(group):
                    return np.mean(group['y_pred'] - group['y_true'])
                pivot = df.groupby(['CANCER_TYPE', atc_col]).apply(calc_bias).unstack(fill_value=np.nan)

            else:
                return None

            return pivot

        except Exception as e:
            logger.error(f"Error calculating heatmap data: {e}")
            return None

    def _get_metric_label(self, metric: str) -> str:
        """Get human-readable label for metric"""
        labels = {
            'mean_ic50': 'Mean LN(IC50)',
            'median_ic50': 'Median LN(IC50)',
            'rmse': 'RMSE',
            'bias': 'Prediction Bias (Pred - True)'
        }
        return labels.get(metric, metric)
