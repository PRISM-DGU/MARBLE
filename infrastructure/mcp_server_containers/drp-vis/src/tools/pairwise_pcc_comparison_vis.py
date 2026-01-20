"""
Pairwise PCC Comparison Visualization
Compare two models' cell-level or drug-level PCC with scatter plot and marginal histograms
"""

import json
import logging
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from utils.plot_style import setup_plot_style, get_model_color
from utils.filename_helper import get_timestamped_filename

logger = logging.getLogger(__name__)


class PairwisePCCComparisonVisualizationTool:
    """Compare two models' PCC at cell or drug level"""

    def __init__(self):
        setup_plot_style()

    def _calculate_group_pcc(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """
        Calculate PCC for each group (cell line or drug)

        Args:
            df: DataFrame with y_true and y_pred
            group_by: Column name to group by ('CELL_LINE_NAME' or 'DRUG_NAME')

        Returns:
            DataFrame with group names and PCC values
        """
        pcc_results = []

        for group_name, group_df in df.groupby(group_by):
            if len(group_df) < 3:  # Need at least 3 points for correlation
                continue

            y_true = group_df['y_true'].values
            y_pred = group_df['y_pred'].values

            pcc, pval = stats.pearsonr(y_true, y_pred)
            pcc_results.append({
                'group': group_name,
                'pcc': pcc,
                'pval': pval,
                'n': len(group_df)
            })

        return pd.DataFrame(pcc_results)

    def apply(self,
              model_a_csv: str,
              model_b_csv: str,
              model_a_name: str,
              model_b_name: str,
              comparison_level: Literal['cell', 'drug'] = 'cell',
              output_dir: str = "figures") -> str:
        """
        Generate pairwise PCC comparison plot with marginal histograms

        Args:
            model_a_csv: Path to model A CSV file
            model_b_csv: Path to model B CSV file
            model_a_name: Name of model A (e.g., "deepdr")
            model_b_name: Name of model B (e.g., "deeptta")
            comparison_level: 'cell' for cell-level or 'drug' for drug-level PCC
            output_dir: Output directory for saving PNG (default: "figures")

        Returns:
            JSON string with status, saved file path, and comparison statistics
        """
        try:
            # Validate files
            if not os.path.exists(model_a_csv):
                return json.dumps({
                    "status": "error",
                    "message": f"Model A CSV not found: {model_a_csv}"
                })

            if not os.path.exists(model_b_csv):
                return json.dumps({
                    "status": "error",
                    "message": f"Model B CSV not found: {model_b_csv}"
                })

            # Load data
            df_a = pd.read_csv(model_a_csv)
            df_b = pd.read_csv(model_b_csv)

            required_cols = ['CELL_LINE_NAME', 'DRUG_NAME', 'y_true', 'y_pred']
            for df, model in [(df_a, model_a_name), (df_b, model_b_name)]:
                if not all(col in df.columns for col in required_cols):
                    return json.dumps({
                        "status": "error",
                        "message": f"Model {model} CSV must have columns: {required_cols}"
                    })

            # Determine grouping column
            group_col = 'CELL_LINE_NAME' if comparison_level == 'cell' else 'DRUG_NAME'
            level_name = 'Cell Line' if comparison_level == 'cell' else 'Drug'

            # Calculate PCC for each group
            pcc_a = self._calculate_group_pcc(df_a, group_col)
            pcc_b = self._calculate_group_pcc(df_b, group_col)

            # Check if PCC results are empty (insufficient data per group)
            if len(pcc_a) == 0 or len(pcc_b) == 0:
                return json.dumps({
                    "status": "error",
                    "message": f"Insufficient data for {comparison_level}-level PCC. "
                               f"Each {group_col} needs at least 3 samples. "
                               f"Model A groups with enough data: {len(pcc_a)}, "
                               f"Model B groups with enough data: {len(pcc_b)}. "
                               f"Try using comparison_level='drug' instead of 'cell'."
                })

            # Merge on common groups
            pcc_merged = pd.merge(
                pcc_a[['group', 'pcc']],
                pcc_b[['group', 'pcc']],
                on='group',
                suffixes=('_a', '_b')
            )

            if len(pcc_merged) == 0:
                return json.dumps({
                    "status": "error",
                    "message": "No common groups found between models"
                })

            pcc_a_values = pcc_merged['pcc_a'].values
            pcc_b_values = pcc_merged['pcc_b'].values

            # Calculate statistics
            mean_pcc_a = np.mean(pcc_a_values)
            mean_pcc_b = np.mean(pcc_b_values)
            win_rate_a = np.sum(pcc_a_values > pcc_b_values) / len(pcc_a_values) * 100
            win_rate_b = np.sum(pcc_b_values > pcc_a_values) / len(pcc_b_values) * 100

            # Create figure with GridSpec for marginal histograms
            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05)

            # Main scatter plot
            ax_main = fig.add_subplot(gs[1:, :-1])
            # Top histogram
            ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            # Right histogram
            ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

            # Main scatter plot
            color_a = get_model_color(model_a_name)
            ax_main.scatter(pcc_a_values, pcc_b_values, alpha=0.6, s=80,
                          color=color_a, edgecolors='black', linewidth=0.8)

            # Diagonal line (y=x)
            min_val = min(pcc_a_values.min(), pcc_b_values.min())
            max_val = max(pcc_a_values.max(), pcc_b_values.max())
            ax_main.plot([min_val, max_val], [min_val, max_val],
                        'k--', linewidth=2, alpha=0.5, label='y=x')

            # Labels
            ax_main.set_xlabel(f'{model_a_name.upper()} PCC', fontsize=14)
            ax_main.set_ylabel(f'{model_b_name.upper()} PCC', fontsize=14)
            ax_main.grid(True, alpha=0.3)

            # Statistics text box
            stats_text = (f'Mean PCC:\n'
                         f'  {model_a_name}: {mean_pcc_a:.3f}\n'
                         f'  {model_b_name}: {mean_pcc_b:.3f}\n\n'
                         f'Win Rate:\n'
                         f'  {model_a_name}: {win_rate_a:.1f}%\n'
                         f'  {model_b_name}: {win_rate_b:.1f}%')

            ax_main.text(0.05, 0.95, stats_text,
                        transform=ax_main.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.9, edgecolor='black', linewidth=1.5))

            # Top histogram (Model A)
            ax_top.hist(pcc_a_values, bins=20, color=color_a, alpha=0.7, edgecolor='black')
            ax_top.axvline(mean_pcc_a, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pcc_a:.3f}')
            ax_top.set_ylabel('Count', fontsize=10)
            ax_top.legend(loc='upper right', fontsize=8)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # Right histogram (Model B)
            color_b = get_model_color(model_b_name)
            ax_right.hist(pcc_b_values, bins=20, color=color_b, alpha=0.7,
                         edgecolor='black', orientation='horizontal')
            ax_right.axhline(mean_pcc_b, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pcc_b:.3f}')
            ax_right.set_xlabel('Count', fontsize=10)
            ax_right.legend(loc='upper right', fontsize=8)
            plt.setp(ax_right.get_yticklabels(), visible=False)

            # Main title
            fig.suptitle(f'{level_name}-Level PCC Comparison: {model_a_name.upper()} vs {model_b_name.upper()}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            base_filename = f"cmp_pcc_{comparison_level}line_{model_a_name}_vs_{model_b_name}" if comparison_level == 'cell' else f"cmp_pcc_{comparison_level}_{model_a_name}_vs_{model_b_name}"
            filename = get_timestamped_filename(base_filename, "png")
            output_path = os.path.join(output_dir, filename)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "statistics": {
                    "model_a_name": model_a_name,
                    "model_b_name": model_b_name,
                    "mean_pcc_a": float(mean_pcc_a),
                    "mean_pcc_b": float(mean_pcc_b),
                    "win_rate_a": float(win_rate_a),
                    "win_rate_b": float(win_rate_b),
                    "n_groups": len(pcc_merged)
                },
                "metadata": {
                    "comparison_level": comparison_level,
                    "level_name": level_name,
                    "output_file": filename
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"Pairwise PCC comparison visualization error: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
