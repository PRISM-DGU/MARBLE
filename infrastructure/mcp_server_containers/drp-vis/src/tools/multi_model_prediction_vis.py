"""
Multi-Model Prediction vs Actual Visualization
Compare 6 DRP models in 3x2 subplot layout (3 rows, 2 columns)
"""

import json
import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils.plot_style import setup_plot_style, get_model_color
from utils.filename_helper import get_timestamped_filename

logger = logging.getLogger(__name__)


class MultiModelPredictionVisualizationTool:
    """Visualize prediction vs actual for multiple models in 3x2 layout (3 rows, 2 columns)"""

    def __init__(self):
        setup_plot_style()
        # Model order will be determined from provided CSV mapping
        self.model_order: list[str] = []

    def apply(self,
              model_csv_paths: Dict[str, str],
              output_dir: str = "figures") -> str:
        """
        Generate 3x2 subplot (3 rows, 2 columns) comparing 6 models' predictions vs actuals

        Args:
            model_csv_paths: Dictionary of {model_name: csv_file_path}
                            CSV must have columns: CELL_LINE_NAME, DRUG_NAME, y_true, y_pred
                            Example: {"deepdr": "/path/to/deepdr.csv", ...}
            output_dir: Output directory for saving PNG (default: "figures")

        Returns:
            JSON string with status, saved file path, and metrics for all models
        """
        try:
            # Validate input (require exactly 6 models for 2x3 grid)
            if len(model_csv_paths) != 6:
                return json.dumps({
                    "status": "error",
                    "message": f"Expected 6 models, got {len(model_csv_paths)}"
                })

            # Preserve provided ordering to keep plots predictable
            self.model_order = list(model_csv_paths.keys())

            # Load data for all models
            model_data = {}
            for model_name, csv_path in model_csv_paths.items():
                if not os.path.exists(csv_path):
                    return json.dumps({
                        "status": "error",
                        "message": f"CSV file not found: {csv_path}"
                    })

                df = pd.read_csv(csv_path)
                required_cols = ['CELL_LINE_NAME', 'DRUG_NAME', 'y_true', 'y_pred']
                if not all(col in df.columns for col in required_cols):
                    return json.dumps({
                        "status": "error",
                        "message": f"CSV must have columns: {required_cols}"
                    })

                model_data[model_name] = df

            # Create 3x2 subplot (3 rows, 2 columns for 6 models)
            fig, axes = plt.subplots(3, 2, figsize=(14, 21))
            axes = axes.flatten()

            all_metrics = {}

            # Plot each model
            for idx, model_name in enumerate(self.model_order):
                if model_name not in model_data:
                    return json.dumps({
                        "status": "error",
                        "message": f"Model '{model_name}' not found in input"
                    })

                df = model_data[model_name]
                ax = axes[idx]

                y_true = df['y_true'].values
                y_pred = df['y_pred'].values

                # Calculate metrics
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - (np.sum((y_true - y_pred)**2) /
                         np.sum((y_true - np.mean(y_true))**2))
                pcc, _ = stats.pearsonr(y_true, y_pred)
                scc, _ = stats.spearmanr(y_true, y_pred)

                all_metrics[model_name] = {
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "pcc": float(pcc),
                    "scc": float(scc),
                    "n_samples": len(y_true)
                }

                # Hexbin plot with density
                hexbin = ax.hexbin(y_true, y_pred, gridsize=30, cmap='YlOrRd',
                                   mincnt=1, alpha=0.8, edgecolors='black', linewidths=0.2)

                # Add colorbar for density
                cbar = plt.colorbar(hexbin, ax=ax)
                cbar.set_label('Count', fontsize=10)

                # Diagonal line (y=x)
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val],
                       'k--', linewidth=2, label='y=x', alpha=0.7)

                # Labels and title
                ax.set_xlabel('Actual LN(IC50)', fontsize=12)
                ax.set_ylabel('Predicted LN(IC50)', fontsize=12)
                ax.set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold')

                # Metrics text box
                metrics_text = (f'R² = {r2:.3f}\n'
                               f'RMSE = {rmse:.3f}\n'
                               f'PCC = {pcc:.3f}\n'
                               f'SCC = {scc:.3f}')

                ax.text(0.05, 0.95, metrics_text,
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.8, edgecolor='black', linewidth=1.5))

                ax.legend(loc='lower right', fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            filename = get_timestamped_filename("predict_vs_actual", "png")
            output_path = os.path.join(output_dir, filename)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "metrics": all_metrics,
                "metadata": {
                    "models": self.model_order,
                    "n_models": 6,
                    "output_file": filename
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"Multi-model prediction visualization error: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
