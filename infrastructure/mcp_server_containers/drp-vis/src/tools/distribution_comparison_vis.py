"""
Distribution Comparison Visualization
Compare IC50 distributions grouped by cancer type or ATC code
Supports boxplot and violin plot
"""

import json
import logging
import os
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plot_style import setup_plot_style, AUTODRP_COLORS, get_qualitative_colors
from utils.filename_helper import get_timestamped_filename
from utils.atc_parser import parse_atc_level, add_atc_levels_to_dataframe

logger = logging.getLogger(__name__)


class DistributionComparisonVisualizationTool:
    """Compare IC50 distributions by cancer type or ATC code"""

    def __init__(self):
        setup_plot_style()

    def apply(self,
              model_csv_paths: Dict[str, str],
              group_by: Literal['cancer_type', 'atc'] = 'cancer_type',
              atc_level: int = 4,
              plot_type: Literal['boxplot', 'violin'] = 'boxplot',
              value_type: Literal['y_true', 'y_pred'] = 'y_true',
              output_dir: str = "figures") -> str:
        """
        Generate distribution comparison plot (boxplot or violin)

        Args:
            model_csv_paths: Dictionary of {model_name: csv_file_path}
                            Can be 1-4 models
                            CSV must have: CELL_LINE_NAME, CANCER_TYPE, DRUG_NAME, ATC_CODE, y_true, y_pred
            group_by: Grouping dimension
                     'cancer_type': Group by cancer type
                     'atc': Group by ATC code
            atc_level: ATC hierarchical level (1-5), only used if group_by='atc'
            plot_type: Type of plot
                      'boxplot': Box-and-whisker plot
                      'violin': Violin plot (distribution shape)
            value_type: Which value to visualize
                       'y_true': Experimental values (GDSC2)
                       'y_pred': Model predictions
            output_dir: Output directory for PNG (default: "figures")

        Returns:
            JSON string with status, output path, and statistics
        """
        try:
            # Validate input
            if len(model_csv_paths) == 0:
                return json.dumps({
                    "status": "error",
                    "message": "At least one model CSV is required"
                })

            # Load and combine data from all models
            combined_data = []
            for model_name, csv_path in model_csv_paths.items():
                if not os.path.exists(csv_path):
                    return json.dumps({
                        "status": "error",
                        "message": f"CSV file not found: {csv_path}"
                    })

                df = pd.read_csv(csv_path)
                required_cols = ['CELL_LINE_NAME', 'CANCER_TYPE', 'DRUG_NAME', 'ATC_CODE', 'y_true', 'y_pred']
                if not all(col in df.columns for col in required_cols):
                    return json.dumps({
                        "status": "error",
                        "message": f"CSV must have columns: {required_cols}"
                    })

                df['MODEL'] = model_name
                combined_data.append(df)

            df_all = pd.concat(combined_data, ignore_index=True)

            # Add ATC level column if grouping by ATC
            if group_by == 'atc':
                df_all = add_atc_levels_to_dataframe(df_all, atc_column='ATC_CODE', levels=[atc_level])
                group_col = f'ATC_LEVEL_{atc_level}'
                group_label = f'ATC Code (Level {atc_level})'
            else:
                group_col = 'CANCER_TYPE'
                group_label = 'Cancer Type'

            # Determine plot layout based on number of models
            n_models = len(model_csv_paths)
            if n_models == 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                axes = [ax]
            elif n_models == 2:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            elif n_models == 3:
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            else:  # 4 models
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()

            # Get colors for groups
            unique_groups = sorted(df_all[group_col].dropna().unique())
            n_groups = len(unique_groups)
            colors = get_qualitative_colors(n_groups)
            palette = dict(zip(unique_groups, colors))

            all_stats = {}

            # Plot each model
            for idx, (model_name, csv_path) in enumerate(model_csv_paths.items()):
                ax = axes[idx] if n_models > 1 else axes[0]
                model_df = df_all[df_all['MODEL'] == model_name]

                # Select value column
                value_col = value_type
                value_label = 'GDSC2 (Experimental)' if value_type == 'y_true' else f'{model_name.upper()} (Predicted)'

                # Create plot
                if plot_type == 'boxplot':
                    sns.boxplot(data=model_df, x=group_col, y=value_col,
                               palette=palette, ax=ax,
                               boxprops=dict(alpha=0.7),
                               flierprops=dict(marker='o', markersize=4, alpha=0.5))
                else:  # violin
                    sns.violinplot(data=model_df, x=group_col, y=value_col,
                                  palette=palette, ax=ax, inner='box', alpha=0.7)

                # Labels and title
                ax.set_xlabel(group_label, fontsize=12, fontweight='bold')
                ax.set_ylabel('LN(IC50)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name.upper()} - {value_label}',
                           fontsize=14, fontweight='bold', pad=10)

                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                # Grid
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                # Calculate statistics
                group_stats = {}
                for group in unique_groups:
                    group_data = model_df[model_df[group_col] == group][value_col].dropna()
                    if len(group_data) > 0:
                        group_stats[group] = {
                            'n': len(group_data),
                            'mean': float(group_data.mean()),
                            'median': float(group_data.median()),
                            'std': float(group_data.std()),
                            'min': float(group_data.min()),
                            'max': float(group_data.max())
                        }

                all_stats[model_name] = group_stats

            # Overall title
            plot_type_name = 'Boxplot' if plot_type == 'boxplot' else 'Violin Plot'
            title = f'IC50 Distribution Comparison by {group_label} - {plot_type_name}'
            if len(model_csv_paths) > 1:
                title += f' ({len(model_csv_paths)} Models)'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            group_suffix = group_by if group_by == 'cancer_type' else f'atc{atc_level}'
            base_filename = f"distribution_{plot_type}_{group_suffix}_{'_'.join(model_csv_paths.keys())}"
            filename = get_timestamped_filename(base_filename, "png")
            output_path = os.path.join(output_dir, filename)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "group_by": group_by,
                "atc_level": atc_level if group_by == 'atc' else None,
                "plot_type": plot_type,
                "value_type": value_type,
                "statistics": all_stats,
                "metadata": {
                    "n_models": len(model_csv_paths),
                    "models": list(model_csv_paths.keys()),
                    "n_groups": n_groups,
                    "groups": unique_groups,
                    "output_file": filename
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"Distribution comparison visualization error: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
