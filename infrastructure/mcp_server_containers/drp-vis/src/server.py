#!/usr/bin/env python
"""
DRP VIS MCP Server
Drug Response Prediction Visualization MCP Server

All visualizations output in English with base64 PNG format.
"""

import itertools
import json
import logging
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd

from fastmcp import FastMCP
from tools.multi_model_prediction_vis import MultiModelPredictionVisualizationTool
from tools.pairwise_pcc_comparison_vis import PairwisePCCComparisonVisualizationTool
from tools.ic50_comparison_vis import IC50ComparisonVisualizationTool
from tools.cancer_atc_heatmap_vis import CancerATCHeatmapVisualizationTool
from tools.distribution_comparison_vis import DistributionComparisonVisualizationTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="drp-vis",
    version="1.0.0"
)

# Initialize tool instances
multi_model_prediction_tool = MultiModelPredictionVisualizationTool()
pairwise_pcc_tool = PairwisePCCComparisonVisualizationTool()
ic50_comparison_tool = IC50ComparisonVisualizationTool()
cancer_atc_heatmap_tool = CancerATCHeatmapVisualizationTool()
distribution_comparison_tool = DistributionComparisonVisualizationTool()


@mcp.tool()
def compare_multi_model_predictions(
    model_csv_paths: Dict[str, str],
    output_dir: str = "figures"
) -> str:
    """
    Compare 6 DRP models' predictions vs actuals in 2x3 subplot layout

    Args:
        model_csv_paths: Dictionary of {model_name: csv_file_path}
                        Must include 6 models (e.g., deepdr, deeptta, stagate, deepst, debate, linear)
                        CSV must have columns: CELL_LINE_NAME, DRUG_NAME, y_true, y_pred
                        Example: {"deepdr": "/path/to/deepdr.csv", ...}
        output_dir: Output directory for saving PNG (default: "figures")

    Returns:
        JSON string with status, saved file path, and metrics for all models
        File saved as: figures/predict_vs_actual_MMDD-HHMM.png
    """
    try:
        return multi_model_prediction_tool.apply(
            model_csv_paths=model_csv_paths,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Multi-model prediction comparison error: {e}")
        return f'{{"error": "Multi-model prediction comparison failed: {str(e)}"}}'


@mcp.tool()
def compare_pairwise_model_pcc(
    model_a_csv: str,
    model_b_csv: str,
    model_a_name: str,
    model_b_name: str,
    comparison_level: str = "cell",
    output_dir: str = "figures"
) -> str:
    """
    Compare two models' cell-level or drug-level PCC with scatter plot and marginal histograms

    Args:
        model_a_csv: Path to model A CSV file
        model_b_csv: Path to model B CSV file
        model_a_name: Name of model A (e.g., "deepdr")
        model_b_name: Name of model B (e.g., "deeptta")
        comparison_level: 'cell' for cell-level or 'drug' for drug-level PCC
        output_dir: Output directory for saving PNG (default: "figures")

    Returns:
        JSON string with status, saved file path, and comparison statistics
        File saved as: figures/cmp_pcc_{level}_{modelA}_vs_{modelB}_MMDD-HHMM.png
    """
    try:
        return pairwise_pcc_tool.apply(
            model_a_csv=model_a_csv,
            model_b_csv=model_b_csv,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            comparison_level=comparison_level,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Pairwise PCC comparison error: {e}")
        return f'{{"error": "Pairwise PCC comparison failed: {str(e)}"}}'


@mcp.tool()
def compare_ic50_by_drug(
    model_csv_paths: Dict[str, str],
    top_n_drugs: Optional[int] = None,
    atc_level: Optional[int] = None,
    output_dir: str = "figures"
) -> str:
    """
    Compare IC50 distributions by drug for 4 models in 2x2 layout (GDSC2 vs Model predictions)

    Args:
        model_csv_paths: Dictionary of {model_name: csv_file_path}
                        Must include all 4 models: deepdr, deeptta, stagate, deepst
                        CSV must have columns: CELL_LINE_NAME, DRUG_NAME, ATC_CODE, y_true, y_pred
                        y_true = GDSC2 (experimental), y_pred = model prediction
                        Example: {"deepdr": "/path/to/deepdr.csv", ...}
        top_n_drugs: Display only top N drugs by sample count (None = display all)
        atc_level: Group by ATC code level (1-5). None = individual drugs
                  1=Anatomical, 2=Therapeutic, 3=Pharmacological, 4=Chemical, 5=Substance
        output_dir: Output directory for saving PNG (default: "figures")

    Returns:
        JSON string with status, saved file path, and statistics for each model and drug
        File saved as: figures/ic50_comparison_4models_MMDD-HHMM.png
    """
    try:
        return ic50_comparison_tool.apply(
            model_csv_paths=model_csv_paths,
            top_n_drugs=top_n_drugs,
            atc_level=atc_level,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"IC50 comparison error: {e}")
        return f'{{"error": "IC50 comparison failed: {str(e)}"}}'


@mcp.tool()
def visualize_cancer_atc_heatmap(
    model_csv_path: str,
    model_name: str,
    metric: str = "mean_ic50",
    atc_level: int = 4,
    value_type: str = "y_true",
    output_dir: str = "figures"
) -> str:
    """
    Visualize Cancer Type × ATC Code heatmap for IC50 metrics

    Args:
        model_csv_path: Path to model CSV file
                       CSV must have: CELL_LINE_NAME, CANCER_TYPE, DRUG_NAME, ATC_CODE, y_true, y_pred
        model_name: Model name for title (e.g., "deepdr")
        metric: Metric to visualize: 'mean_ic50', 'median_ic50', 'rmse', 'bias'
        atc_level: ATC hierarchical level (1-5, default: 4)
                  1=Anatomical, 2=Therapeutic, 3=Pharmacological, 4=Chemical, 5=Substance
        value_type: Which value to use: 'y_true' (GDSC2), 'y_pred' (model), 'both'
        output_dir: Output directory for saving PNG (default: "figures")

    Returns:
        JSON string with status, saved file path, and heatmap statistics
        File saved as: figures/heatmap_{metric}_{model}_atc{level}_MMDD-HHMM.png
    """
    try:
        return cancer_atc_heatmap_tool.apply(
            model_csv_path=model_csv_path,
            model_name=model_name,
            metric=metric,
            atc_level=atc_level,
            value_type=value_type,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Cancer-ATC heatmap error: {e}")
        return f'{{"error": "Cancer-ATC heatmap failed: {str(e)}"}}'


@mcp.tool()
def compare_distribution_by_group(
    model_csv_paths: Dict[str, str],
    group_by: str = "cancer_type",
    atc_level: int = 4,
    plot_type: str = "boxplot",
    value_type: str = "y_true",
    output_dir: str = "figures"
) -> str:
    """
    Compare IC50 distributions grouped by cancer type or ATC code

    Args:
        model_csv_paths: Dictionary of {model_name: csv_file_path}
                        Can be 1-4 models
                        CSV must have: CELL_LINE_NAME, CANCER_TYPE, DRUG_NAME, ATC_CODE, y_true, y_pred
                        Example: {"deepdr": "/path/to/deepdr.csv", ...}
        group_by: Grouping dimension: 'cancer_type' or 'atc'
        atc_level: ATC hierarchical level (1-5, default: 4), only used if group_by='atc'
        plot_type: Type of plot: 'boxplot' or 'violin'
        value_type: Which value to visualize: 'y_true' (GDSC2) or 'y_pred' (model)
        output_dir: Output directory for saving PNG (default: "figures")

    Returns:
        JSON string with status, saved file path, and distribution statistics
        File saved as: figures/distribution_{plot_type}_{group_by}_MMDD-HHMM.png
    """
    try:
        return distribution_comparison_tool.apply(
            model_csv_paths=model_csv_paths,
            group_by=group_by,
            atc_level=atc_level,
            plot_type=plot_type,
            value_type=value_type,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Distribution comparison error: {e}")
        return f'{{"error": "Distribution comparison failed: {str(e)}"}}'


@mcp.tool()
def visualize_merge_results(
    merge_csv_path: str = "/workspace/experiments/test_result/Merge_result.csv",
    output_dir: str = "/workspace/experiments/test_result/visualization",
    comparison_level: str = "drug"
) -> str:
    """
    Generate visualizations from a merged prediction CSV.

    The merged CSV must include columns:
    - cell_line_name
    - drug_name
    - true (ground truth LN(IC50))
    - one column per model prediction (e.g., deepdr, deeptta, stagate, deepst)

    This tool will:
    1) Split the merged CSV into per-model CSVs with columns: CELL_LINE_NAME, DRUG_NAME, y_true, y_pred
    2) Run pairwise PCC visualizations for ALL model pairs
    3) Run multi-model prediction vs actual (2x2) when 4+ models are available
    4) Save all PNGs to output_dir (default: /workspace/experiments/test_result/visualization)

    Returns JSON with generated file paths and per-plot metadata.
    """

    try:
        if comparison_level not in {"cell", "drug"}:
            return json.dumps({
                "status": "error",
                "message": f"Invalid comparison_level: {comparison_level}. Use 'cell' or 'drug'."
            })

        if not os.path.exists(merge_csv_path):
            return json.dumps({
                "status": "error",
                "message": f"Merge CSV not found: {merge_csv_path}"
            })

        df = pd.read_csv(merge_csv_path)
        df.columns = [c.lower() for c in df.columns]

        required_cols = {"cell_line_name", "drug_name", "true"}
        if not required_cols.issubset(set(df.columns)):
            return json.dumps({
                "status": "error",
                "message": f"Merge CSV must include columns: {sorted(required_cols)}"
            })

        # Identify prediction columns (everything except base columns)
        prediction_cols = [c for c in df.columns if c not in required_cols]

        if len(prediction_cols) < 2:
            return json.dumps({
                "status": "error",
                "message": "Need at least 2 model prediction columns for visualization"
            })

        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, "tmp_csv")
        os.makedirs(tmp_dir, exist_ok=True)

        model_csv_paths: Dict[str, str] = {}
        for col in prediction_cols:
            model_name = col.lower()
            model_df = pd.DataFrame({
                "CELL_LINE_NAME": df["cell_line_name"],
                "DRUG_NAME": df["drug_name"],
                "y_true": df["true"],
                "y_pred": df[col]
            })

            csv_path = os.path.join(tmp_dir, f"{model_name}.csv")
            model_df.to_csv(csv_path, index=False)
            model_csv_paths[model_name] = csv_path

        generated_files: List[str] = []
        pairwise_outputs: List[Dict[str, str]] = []
        multi_model_output: Optional[Dict[str, str]] = None

        # Pairwise PCC for all combinations
        for (model_a, path_a), (model_b, path_b) in itertools.combinations(model_csv_paths.items(), 2):
            result_json = pairwise_pcc_tool.apply(
                model_a_csv=path_a,
                model_b_csv=path_b,
                model_a_name=model_a,
                model_b_name=model_b,
                comparison_level=comparison_level,
                output_dir=output_dir
            )

            try:
                parsed = json.loads(result_json)
            except Exception:
                parsed = {"status": "error", "raw": result_json}

            if parsed.get("status") == "success" and parsed.get("output_path"):
                generated_files.append(parsed["output_path"])
            pairwise_outputs.append({
                "models": f"{model_a} vs {model_b}",
                "result": parsed
            })

        # Multi-model (2x3) when 6 models are available
        if len(model_csv_paths) >= 6:
            ordered_paths = {name: model_csv_paths[name] for name in list(model_csv_paths.keys())[:6]}
            multi_json = multi_model_prediction_tool.apply(
                model_csv_paths=ordered_paths,
                output_dir=output_dir
            )

            try:
                parsed_multi = json.loads(multi_json)
            except Exception:
                parsed_multi = {"status": "error", "raw": multi_json}

            if parsed_multi.get("status") == "success" and parsed_multi.get("output_path"):
                generated_files.append(parsed_multi["output_path"])

            multi_model_output = parsed_multi

        # IC50 comparison (6 models, drug-level boxplot) when 6 models are available
        ic50_output: Optional[Dict[str, str]] = None
        if len(model_csv_paths) >= 6:
            ordered_paths = {name: model_csv_paths[name] for name in list(model_csv_paths.keys())[:6]}
            ic50_json = ic50_comparison_tool.apply(
                model_csv_paths=ordered_paths,
                top_n_drugs=20,  # Top 20 drugs by sample count
                atc_level=None,  # No ATC grouping
                output_dir=output_dir
            )

            try:
                parsed_ic50 = json.loads(ic50_json)
            except Exception:
                parsed_ic50 = {"status": "error", "raw": ic50_json}

            if parsed_ic50.get("status") == "success" and parsed_ic50.get("output_path"):
                generated_files.append(parsed_ic50["output_path"])

            ic50_output = parsed_ic50

        response = {
            "status": "success",
            "merge_csv_path": merge_csv_path,
            "output_dir": output_dir,
            "generated_files": generated_files,
            "pairwise": pairwise_outputs,
            "multi_model": multi_model_output,
            "ic50_comparison": ic50_output,
            "models_detected": list(model_csv_paths.keys()),
            "comparison_level": comparison_level
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Merge visualization failed: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    logger.info("DRP VIS MCP Server initialized")
    logger.info("Available tools (5 active):")
    logger.info("  ✅ compare_multi_model_predictions - 6 models prediction vs actual (2x3 hexbin)")
    logger.info("  ✅ compare_pairwise_model_pcc - 2 models PCC comparison (cell/drug level)")
    logger.info("  ✅ compare_ic50_by_drug - 4 models IC50 distribution (2x2 grouped boxplot)")
    logger.info("  ✅ visualize_cancer_atc_heatmap - Cancer type × ATC code heatmap")
    logger.info("  ✅ compare_distribution_by_group - Distribution comparison by cancer/ATC")

    # Start the MCP server
    logger.info("Starting MCP server...")
    mcp.run()
