"""Analysis Subgraph - Direct Visualization Execution.

This subgraph directly executes DRP-VIS visualization tools via Docker exec,
bypassing MCP agent issues in langgraph dev environment. It now supports:
- Visualization from merged results (existing)
- Unified HTML report (metrics + D3 scatter + embedded Mermaid diagrams)
"""

import json
import os
import subprocess
import textwrap
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage

from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.utils import get_user_id

# Module-level compiled subgraph
ANALYSIS_SUBGRAPH = None


def _run_visualization(state: MARBLEState) -> Dict[str, Any]:
    """Execute DRP-VIS visualization directly via Docker exec."""

    user_id = get_user_id()
    container_name = f"drp-vis_{user_id}"

    merge_csv = "/workspace/experiments/test_result/Merge_result.csv"
    output_dir = "/workspace/experiments/test_result/visualization"

    logger.info(f"🎨 [VISUALIZATION] Running visualization via {container_name}...")

    # Python script to run inside container
    script = f'''
import sys, os, json
sys.path.insert(0, '/app/src')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('/tmp/matplotlib', exist_ok=True)

from server import visualize_merge_results
result = visualize_merge_results.fn(
    merge_csv_path="{merge_csv}",
    output_dir="{output_dir}",
    comparison_level="drug"
)
print(result)
'''

    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            logger.error(f"❌ [VISUALIZATION] Docker exec failed: {error_msg}")
            return {
                "messages": [AIMessage(content=f"Visualization failed: {error_msg}")],
                "processing_logs": state.get("processing_logs", []) + [
                    f"[VISUALIZATION] Error: {error_msg}"
                ]
            }

        # Parse result JSON
        output = result.stdout.strip()
        try:
            viz_result = json.loads(output)
        except json.JSONDecodeError:
            viz_result = {"status": "unknown", "raw_output": output}

        # Build response message
        if viz_result.get("status") == "success":
            files = viz_result.get("generated_files", [])
            msg = f"✅ Visualization complete! Generated {len(files)} files:\n"
            for f in files:
                msg += f"  - {f}\n"
        else:
            msg = f"Visualization result: {json.dumps(viz_result, indent=2)}"

        logger.info(f"✅ [VISUALIZATION] Complete: {len(viz_result.get('generated_files', []))} files")

        return {
            "messages": [AIMessage(content=msg)],
            "processing_logs": state.get("processing_logs", []) + [
                f"[VISUALIZATION] Generated {len(viz_result.get('generated_files', []))} files"
            ]
        }

    except subprocess.TimeoutExpired:
        logger.error("❌ [VISUALIZATION] Timeout after 120s")
        return {
            "messages": [AIMessage(content="Visualization timed out after 120 seconds")],
            "processing_logs": state.get("processing_logs", []) + [
                "[VISUALIZATION] Timeout"
            ]
        }
    except Exception as e:
        logger.error(f"❌ [VISUALIZATION] Exception: {e}")
        return {
            "messages": [AIMessage(content=f"Visualization error: {str(e)}")],
            "processing_logs": state.get("processing_logs", []) + [
                f"[VISUALIZATION] Exception: {str(e)}"
            ]
        }

def _run_make_html(state: MARBLEState) -> Dict[str, Any]:
    """Generate HTML analysis report (data split, hyperparams, metrics, scatter)."""

    user_id = get_user_id()
    container_name = f"drp-vis_{user_id}"

    base_dir = "/workspace/experiments/test_result"
    merge_csv = f"{base_dir}/Merge_result.csv"
    metrics_csv = f"{base_dir}/model_comparison_results_only_random.csv"
    data_split_yaml = f"{base_dir}/debate_data_split.yaml"
    hyper_yaml = f"{base_dir}/debate_hyperparameter.yaml"
    output_dir = f"{base_dir}/visualization"
    output_html = f"{output_dir}/analysis_result.html"

    logger.info(f"🧾 [HTML] Generating analysis HTML (with Mermaid) via {container_name}...")

    # Pass OpenAI credentials/model into the container for LLM summary generation.
    env_flags = []
    openai_key = os.getenv("OPENAI_API_KEY")
    summary_model = os.getenv("SUMMARY_MODEL")
    if openai_key:
        env_flags.extend(["-e", f"OPENAI_API_KEY={openai_key}"])
    if summary_model:
        env_flags.extend(["-e", f"SUMMARY_MODEL={summary_model}"])

    script = textwrap.dedent('''\
import json, os, math, textwrap, sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr
os.makedirs("{output_dir}", exist_ok=True)

def read_yaml_table(path, key):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {{}}
    return data.get(key, [])

def compute_stats(df):
    true = df["true"]
    pred = df["debate"]
    rmse = math.sqrt(((pred - true) ** 2).mean())
    pcc = true.corr(pred)
    scc = spearmanr(true, pred, nan_policy="omit").statistic
    return rmse, pcc, scc


def load_yaml_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def read_text_safe(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None


def summarize_with_llm(plan, report, agenda, linear_cfg, cell_cfg, metrics_dict):
    """Attempt comprehensive LLM summary; fallback to compact bullets."""
    merged_parts = []

    # Document descriptions for context
    if agenda:
        merged_parts.append("=== AGENDA (Improvement Direction & Experiment Plan for Linear Model) ===\\n" + agenda[:4000])
    if report:
        merged_parts.append("=== FINAL DEBATE REPORT (Feasibility & Risk Assessment for Linear Model Improvement) ===\\n" + report[:4000])
    if plan:
        merged_parts.append("=== IMPLEMENTATION PLAN (Technical Evaluation of Implementation Feasibility & Risks) ===\\n" + plan[:4000])

    # YAML config comparison
    if linear_cfg:
        import json
        merged_parts.append("=== BASELINE LINEAR MODEL STRUCTURE (linear.yaml) ===\\n" + json.dumps(linear_cfg, indent=2)[:2000])
    if cell_cfg:
        import json
        merged_parts.append("=== IMPROVED MODEL STRUCTURE (improved_config_cell_encoder.yaml) ===\\n" + json.dumps(cell_cfg, indent=2)[:2000])

    # Performance metrics comparison
    if metrics_dict:
        metrics_text = f"""=== PERFORMANCE COMPARISON (linear=before, debate=after) ===
- RMSE: {metrics_dict.get('rmse_before', 'N/A')} → {metrics_dict.get('rmse_after', 'N/A')} (improvement: {metrics_dict.get('rmse_improvement', 'N/A')})
- PCC (Pearson): {metrics_dict.get('pcc_before', 'N/A')} → {metrics_dict.get('pcc_after', 'N/A')} (improvement: {metrics_dict.get('pcc_improvement', 'N/A')})
- SCC (Spearman): {metrics_dict.get('scc_before', 'N/A')} → {metrics_dict.get('scc_after', 'N/A')} (improvement: {metrics_dict.get('scc_improvement', 'N/A')})"""
        merged_parts.append(metrics_text)

    merged_text = "\\n\\n".join(merged_parts)

    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and merged_text:
            model_name = os.getenv("SUMMARY_MODEL", "gpt-5.1")
            client = openai.OpenAI(api_key=api_key)
            system_prompt = (
                "You are an expert technical summarizer for drug response prediction model improvements.\\n\\n"
                "Return EXACTLY 4 bullet points with these labels (no markdown, no asterisks):\\n\\n"
                "1. Objective: [improvement goal and rationale in 1-2 sentences]\\n"
                "2. Architecture: [specific changes - encoder types, dimensions, activations, dropout]\\n"
                "3. Training: [hyperparameter changes - lr, batch size, optimizer, epochs]\\n"
                "4. Performance: [RMSE/PCC/SCC before → after with % improvement]\\n\\n"
                "Format rules:\\n"
                "- Start each line with the label followed by a colon (e.g., 'Objective:')\\n"
                "- Do NOT use asterisks, bullets, or markdown formatting\\n"
                "- Include specific numbers and technical details\\n"
                "- Keep each point concise (1-2 sentences)"
            )
            def _extract_responses_text(resp):
                try:
                    blocks = getattr(resp, "output", None) or []
                    parts = []
                    for block in blocks:
                        for content in getattr(block, "content", None) or []:
                            text_val = getattr(content, "text", None)
                            if text_val:
                                parts.append(text_val)
                            elif isinstance(content, dict) and content.get("text"):
                                parts.append(content["text"])
                    return "".join(parts).strip() if parts else None
                except Exception:
                    return None

            response_text = None

            if model_name.startswith("gpt-5") and hasattr(client, "responses"):
                try:
                    resp = client.responses.create(
                        model=model_name,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": merged_text},
                        ],
                        max_output_tokens=800,
                    )
                    response_text = _extract_responses_text(resp)
                except Exception as e:
                    print(f"[SUMMARY] OpenAI responses.create failed ({model_name}): {{type(e).__name__}}: {{e}}", file=sys.stderr)

            if not response_text:
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": merged_text}
                        ],
                        temperature=0.2,
                        max_tokens=800
                    )
                    response_text = completion.choices[0].message.content.strip()
                except Exception as e:
                    print(f"[SUMMARY] OpenAI chat.completions failed ({model_name}): {{type(e).__name__}}: {{e}}", file=sys.stderr)

            if response_text:
                return response_text, True
    except Exception as e:
        print(f"[SUMMARY] OpenAI API failed: {{type(e).__name__}}: {{e}}", file=sys.stderr)

    # Enhanced fallback: extract actual content from files with fixed labels
    import re
    bullets = {"objective": None, "architecture": None, "training": None, "performance": None}

    def truncate_smart(text, max_len):
        """Truncate text at word boundary."""
        if len(text) <= max_len:
            return text
        truncated = text[:max_len].rsplit(' ', 1)[0]
        return truncated.rstrip('.,;:`') + "..."

    # 1. Extract improvement target and direction from agenda
    if agenda:
        target_match = re.search(r"##\\s*Target Component\\s*\\n+([^\\n#]+)", agenda)
        target = target_match.group(1).strip() if target_match else None

        limitation_match = re.search(r"\\*\\*Limitation:\\*\\*\\s*([^\\n]+)", agenda)
        limitation = limitation_match.group(1).strip() if limitation_match else None

        obj_bits = []
        if target:
            obj_bits.append(f"Target: {target}")
        if limitation:
            obj_bits.append(truncate_smart(limitation, 150))
        if obj_bits:
            bullets["objective"] = "; ".join(obj_bits)

    # 2. Extract feasibility and risks from debate report
    if report:
        feasibility_match = re.search(r"Implementation Feasibility:\\s*([^\\n]+(?:\\n[^\\n#*]+)*)", report)
        if feasibility_match:
            feas_text = truncate_smart(feasibility_match.group(1).strip(), 200)
            bullets["objective"] = (bullets["objective"] + ". " if bullets["objective"] else "") + feas_text

    # 3. Architecture change details
    if linear_cfg and cell_cfg:
        old_cell = linear_cfg.get("model", {}).get("cell_encoder", {}).get("type", "linear")
        new_cell = cell_cfg.get("model", {}).get("cell_encoder", {}).get("type", "unknown")
        new_cell_cfg = cell_cfg.get("model", {}).get("cell_encoder", {})
        hidden = new_cell_cfg.get("hidden_dims", [])
        dropout = new_cell_cfg.get("dropout", 0)
        arch_parts = [f"cell_encoder: {old_cell} to {new_cell}"]
        if hidden:
            arch_parts.append(f"hidden_dims={hidden}")
        if dropout:
            arch_parts.append(f"dropout={dropout}")
        bullets["architecture"] = ", ".join(arch_parts)

    # 4. Training hyperparameters
    if cell_cfg:
        train_cfg = cell_cfg.get("training", {})
        lr = train_cfg.get("learning_rate", train_cfg.get("lr"))
        wd = train_cfg.get("weight_decay")
        bs = train_cfg.get("batch_size")
        epochs = train_cfg.get("epochs")
        train_parts = []
        if lr:
            train_parts.append(f"lr={lr}")
        if wd:
            train_parts.append(f"weight_decay={wd}")
        if bs:
            train_parts.append(f"batch_size={bs}")
        if epochs:
            train_parts.append(f"epochs={epochs}")
        if train_parts:
            bullets["training"] = ", ".join(train_parts)

    # 5. Performance metrics
    if metrics_dict:
        perf_parts = [
            f"RMSE: {metrics_dict.get('rmse_before', '?')} to {metrics_dict.get('rmse_after', '?')} ({metrics_dict.get('rmse_improvement', 'N/A')})",
            f"PCC: {metrics_dict.get('pcc_before', '?')} to {metrics_dict.get('pcc_after', '?')}",
            f"SCC: {metrics_dict.get('scc_before', '?')} to {metrics_dict.get('scc_after', '?')}",
        ]
        bullets["performance"] = "; ".join(perf_parts)

    obj_val = bullets["objective"] or "Improve model architecture for better drug response prediction."
    arch_val = bullets["architecture"] or "No architecture changes detected."
    train_val = bullets["training"] or "Default training configuration."
    perf_val = bullets["performance"] or "No performance metrics available."

    fallback_text = "Objective: " + obj_val + "\\nArchitecture: " + arch_val + "\\nTraining: " + train_val + "\\nPerformance: " + perf_val
    return fallback_text, False


def encoder_label(enc_cfg, role):
    enc_type = enc_cfg.get("type", "unknown")
    input_dim = enc_cfg.get("input_dim", "?")
    output_dim = enc_cfg.get("output_dim", "?")
    hidden = enc_cfg.get("hidden_dims")
    lines = [f"{{role}} Encoder: {{enc_type}}"]
    if hidden:
        lines.append(f"dims: {{input_dim}} → {{hidden}} → {{output_dim}}")
    else:
        lines.append(f"dims: {{input_dim}} → {{output_dim}}")
    if enc_cfg.get("activation"):
        lines.append(f"act: {{enc_cfg['activation']}}")
    if enc_cfg.get("dropout") is not None:
        lines.append(f"dropout: {{enc_cfg['dropout']}}")
    return "\\n".join(lines)


def build_mermaid(cfg, title):
    model = cfg.get("model", {{}}) or {{}}
    data_cfg = cfg.get("data", {{}}) or {{}}
    loaders = (data_cfg.get("loaders", {{}}) or {{}})

    drug_loader = loaders.get("drug", {{}}).get("type", "drug input")
    cell_loader = loaders.get("cell", {{}}).get("type", "cell input")

    drug_enc = model.get("drug_encoder", {{}}) or {{}}
    cell_enc = model.get("cell_encoder", {{}}) or {{}}
    decoder = model.get("decoder", {{}}) or {{}}

    drug_label = encoder_label(drug_enc, "Drug")
    cell_label = encoder_label(cell_enc, "Cell")
    dec_type = decoder.get("type", "decoder")
    dec_out = decoder.get("output_dim", "?")

    lines = [
        "flowchart TD",
        '    A["Drug Input\\n' + drug_loader + '"]',
        '    B["Cell Input\\n' + cell_loader + '"]',
        '    D["' + drug_label + '"]',
        '    E["' + cell_label + '"]',
        '    F["Drug Embedding"]',
        '    G["Cell Embedding"]',
        '    H["Fusion / Concatenate"]',
        '    I["Decoder: ' + dec_type + '\\noutput_dim=' + str(dec_out) + '"]',
        '    J["IC50 Prediction"]',
        "    A --> D",
        "    B --> E",
        "    D --> F",
        "    E --> G",
        "    F --> H",
        "    G --> H",
        "    H --> I",
        "    I --> J",
    ]

    title_block = "%% " + title
    return title_block + "\\n" + "\\n".join(lines)

# Load data
df = pd.read_csv("{merge_csv}")
df = df[["cell_line_name", "drug_name", "true", "linear", "debate"]].dropna()
df["abs_error"] = (df["debate"] - df["true"]).abs()

# Compute stats for improved model (debate)
rmse_data, pcc_data, scc_data = compute_stats(df)

# Compute stats for baseline model (linear) for comparison
def compute_stats_for_column(df, pred_col):
    true = df["true"]
    pred = df[pred_col]
    rmse = math.sqrt(((pred - true) ** 2).mean())
    pcc = true.corr(pred)
    scc = spearmanr(true, pred, nan_policy="omit").statistic
    return rmse, pcc, scc

rmse_linear, pcc_linear, scc_linear = compute_stats_for_column(df, "linear")
rmse_debate, pcc_debate, scc_debate = compute_stats_for_column(df, "debate")

# Calculate improvements (for RMSE: lower is better, for PCC/SCC: higher is better)
rmse_improvement = f"{{rmse_linear - rmse_debate:.4f}} ({{((rmse_linear - rmse_debate) / rmse_linear * 100):.2f}}% reduction)"
pcc_improvement = f"{{pcc_debate - pcc_linear:.4f}} ({{((pcc_debate - pcc_linear) / abs(pcc_linear) * 100):.2f}}% increase)"
scc_improvement = f"{{scc_debate - scc_linear:.4f}} ({{((scc_debate - scc_linear) / abs(scc_linear) * 100):.2f}}% increase)"

metrics_comparison = {{
    "rmse_before": f"{{rmse_linear:.4f}}",
    "rmse_after": f"{{rmse_debate:.4f}}",
    "rmse_improvement": rmse_improvement,
    "pcc_before": f"{{pcc_linear:.4f}}",
    "pcc_after": f"{{pcc_debate:.4f}}",
    "pcc_improvement": pcc_improvement,
    "scc_before": f"{{scc_linear:.4f}}",
    "scc_after": f"{{scc_debate:.4f}}",
    "scc_improvement": scc_improvement,
}}

data_split_rows = read_yaml_table("{data_split_yaml}", "splits")
hyper_rows = read_yaml_table("{hyper_yaml}", "hyperparameters")

# Prepare JSON data for embedding
scatter_data = df.to_dict(orient="records")

def fmt(x):
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{{x:.3f}}"

metric_cards = {{
    "rmse": fmt(rmse_data),
    "pcc": fmt(pcc_data),
    "scc": fmt(scc_data),
}}

footer_stats = {{
    "rmse": fmt(rmse_data),
    "pcc": fmt(pcc_data),
    "scc": fmt(scc_data),
}}

data_split_html = "".join([
    f"<tr><td>{{r.get('split','')}}</td><td>{{r.get('samples','')}}</td><td>{{r.get('percentage','')}}</td></tr>"
    for r in data_split_rows
])

hyper_html = "".join([
    f"<tr><td>{{r.get('parameter','')}}</td><td>{{r.get('value','')}}</td></tr>"
    for r in hyper_rows
])

# Mermaid diagrams
linear_cfg = load_yaml_safe("{base_dir}/linear.yaml")
cell_cfg = load_yaml_safe("{base_dir}/improved_config_cell_encoder.yaml")

mermaid_linear = build_mermaid(linear_cfg, "Linear Baseline") if linear_cfg else "graph TD\\n A[linear.yaml missing] --> B[No diagram]"
mermaid_cell = build_mermaid(cell_cfg, "Improved Cell Encoder") if cell_cfg else "graph TD\\n A[improved_config_cell_encoder.yaml missing] --> B[No diagram]"

# Summary generation from local MD files (LLM with fallback)
plan_text = read_text_safe("{base_dir}/implementation_plan_detailed.md")
report_text = read_text_safe("{base_dir}/final_debate_report.md")
agenda_text = read_text_safe("{base_dir}/agenda.md")

summary_text, summary_llm = summarize_with_llm(plan_text, report_text, agenda_text, linear_cfg, cell_cfg, metrics_comparison)

def format_summary_html(text, max_items=10, max_chars=3000):
    import re
    if not text:
        return "<p class='summary-empty'>No summary available.</p>"
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "..."

    raw_items = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. " or "1) "
        line = re.sub(r"^[0-9]+[.)][ ]*", "", line).strip()
        # Strip leading bullet markers
        line = re.sub(r"^[-*]+[ ]*", "", line).strip()
        # Remove any stray ** markers
        line = line.replace("**", "")
        # Skip empty lines
        if not line:
            continue
        raw_items.append(line)

    if not raw_items:
        raw_items = [cleaned]

    items = raw_items[:max_items]

    # Expected labels
    labels = ["Objective", "Architecture", "Training", "Performance"]

    # Check if first item starts with a known label
    first_lower = items[0].lower() if items else ""
    has_labels = any(first_lower.startswith(lbl.lower()) for lbl in labels)

    if len(items) >= 4 and not has_labels:
        # Force labels if LLM omitted them
        normalized = []
        for i, item in enumerate(items[:4]):
            label = labels[i] if i < len(labels) else "Point " + str(i+1)
            # Remove any existing label prefix
            for lbl in labels:
                if item.lower().startswith(lbl.lower()):
                    item = re.sub(r"^[A-Za-z]+:[ ]*", "", item)
                    break
            normalized.append("<strong>" + label + ":</strong> " + item.strip())
        items = normalized
    else:
        # Convert existing "Label:" to "<strong>Label:</strong>"
        formatted = []
        for item in items:
            matched = False
            for lbl in labels:
                if item.lower().startswith(lbl.lower() + ":"):
                    rest = item[len(lbl)+1:].strip()
                    item = "<strong>" + lbl + ":</strong> " + rest
                    matched = True
                    break
            formatted.append(item)
        items = formatted

    lis = "".join("<li>" + item + "</li>" for item in items)
    return "<ul class='summary-list'>" + lis + "</ul>"

summary_html = format_summary_html(summary_text)

html_template = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Training Configuration &amp; Performance Metrics</title>
  <script src='https://d3js.org/d3.v7.min.js'></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <style>
    :root {{
      --bg: #f3f4ff;
      --card: #ffffff;
      --accent: #6b63ff;
      --accent-light: #867ff5;
      --text: #1f2340;
      --muted: #6d7890;
      --border: #e2e6f5;
      --shadow: 0 10px 30px rgba(79, 70, 229, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      margin: 0;
      padding: 20px;
      color: var(--text);
    }}
    .container {{
      background: var(--card);
      border-radius: 16px;
      padding: 24px;
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      max-width: 1100px;
      margin: 0 auto 28px auto;
    }}
    h2 {{
      margin: 0 0 16px 0;
      font-size: 20px;
      font-weight: 700;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    h2::before {{
      content: "";
      width: 6px;
      height: 20px;
      background: var(--accent);
      border-radius: 3px;
      display: inline-block;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fbfbff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      color: var(--text);
    }}
    th {{
      background: var(--accent);
      color: white;
      text-align: left;
      padding: 8px 10px;
      font-weight: 600;
      font-size: 12px;
    }}
    td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .metric-card {{
      background: linear-gradient(135deg, #6b63ff, #8b7cff);
      color: white;
      border-radius: 12px;
      padding: 14px;
      text-align: center;
      box-shadow: var(--shadow);
    }}
    .metric-title {{
      font-size: 13px;
      margin-bottom: 8px;
      letter-spacing: 0.3px;
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    #scatter {{
      width: 100%;
      height: 420px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: var(--shadow);
    }}
    .legend {{
      position: absolute;
      right: 24px;
      top: 90px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.9);
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: #1f2340;
      color: #fff;
      padding: 10px;
      border-radius: 8px;
      font-size: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }}
    .footer-metrics {{
      margin-top: 8px;
      text-align: right;
      color: var(--muted);
      font-size: 12px;
    }}
    .mermaid-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .mermaid-card {{
      background: #fbfbff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .mermaid-card h3 {{
      margin: 0 0 12px 0;
      color: var(--accent);
      font-size: 15px;
    }}
    .summary-card {{
      background: #fdfdff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .summary-body {{
      color: var(--text);
      line-height: 1.5;
      font-size: 14px;
      white-space: normal;
    }}
    .summary-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--text);
      line-height: 1.5;
      font-size: 14px;
    }}
    .summary-list li {{
      margin-bottom: 6px;
    }}
    .summary-empty {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <h2>Training Configuration &amp; Performance Metrics</h2>
    <div class="grid">
      <div class="card">
        <h3 style="margin: 0 0 10px 0; color: var(--accent); font-size: 14px;">Data Split</h3>
        <table>
          <thead><tr><th>Split</th><th>Samples</th><th>Percentage</th></tr></thead>
          <tbody>
            __DATA_SPLIT__
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3 style="margin: 0 0 10px 0; color: var(--accent); font-size: 14px;">Hyperparameters</h3>
        <table>
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>
            __HYPERPARAM__
          </tbody>
        </table>
      </div>
    </div>
    <div class="metrics">
      <div class="metric-card"><div class="metric-title">RMSE</div><div class="metric-value">__RMSE__</div></div>
      <div class="metric-card"><div class="metric-title">PCC (Pearson)</div><div class="metric-value">__PCC__</div></div>
      <div class="metric-card"><div class="metric-title">SCC (Spearman)</div><div class="metric-value">__SCC__</div></div>
    </div>
  </div>

  <div class="container">
    <h2>Prediction Results Visualization</h2>
    <div id="scatter"></div>
    <div class="footer-metrics">RMSE: __FOOT_RMSE__ | PCC: __FOOT_PCC__ | SCC: __FOOT_SCC__</div>
  </div>

  <div class="container">
    <h2>Model Architecture (Mermaid)</h2>
    <div class="mermaid-grid">
      <div class="mermaid-card">
        <h3>Linear Baseline</h3>
        <div class="mermaid">
__MERMAID_LINEAR__
        </div>
      </div>
      <div class="mermaid-card">
        <h3>Improved Cell Encoder</h3>
        <div class="mermaid">
__MERMAID_CELL__
        </div>
      </div>
    </div>
  </div>

  <div class="container">
    <h2>Summary</h2>
    <div class="mermaid-card summary-card">
      <div class="summary-body">__SUMMARY__</div>
    </div>
  </div>

  <script>
    mermaid.initialize({{startOnLoad:true, theme: "default"}});
    const data = __SCATTER__;
    const width = 980, height = 440;
    const margin = {{top: 30, right: 140, bottom: 60, left: 60}};

    const x = d3.scaleLinear()
      .domain(d3.extent(data, d => d.true)).nice()
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain(d3.extent(data, d => d.debate)).nice()
      .range([height - margin.bottom, margin.top]);

    const minErr = d3.min(data, d => d.abs_error);
    const maxErr = d3.max(data, d => d.abs_error);
    const color = d3.scaleLinear()
      .domain([minErr, (minErr + maxErr) / 2, maxErr])
      .range(["#4f8ef7", "#8e63ff", "#f16b6b"]);

    const svg = d3.select("#scatter")
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    // Axes
    svg.append("g")
      .attr("transform", `translate(0, ${{height - margin.bottom}})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", width / 2)
      .attr("y", 45)
      .attr("fill", "#4a4f6b")
      .attr("text-anchor", "middle")
      .text("Actual IC50 (log μM)");

    svg.append("g")
      .attr("transform", `translate(${{margin.left}}, 0)`)
      .call(d3.axisLeft(y))
      .append("text")
      .attr("x", -height / 2)
      .attr("y", -40)
      .attr("transform", "rotate(-90)")
      .attr("fill", "#4a4f6b")
      .attr("text-anchor", "middle")
      .text("Predicted IC50 (log μM)");

    // Identity line
    const lineData = d3.extent([...data.map(d => d.true), ...data.map(d => d.debate)]);
    svg.append("line")
      .attr("x1", x(lineData[0]))
      .attr("y1", y(lineData[0]))
      .attr("x2", x(lineData[1]))
      .attr("y2", y(lineData[1]))
      .attr("stroke", "#9aa3c3")
      .attr("stroke-dasharray", "5,5")
      .attr("stroke-width", 1.5);

    // Tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "tooltip")
      .style("opacity", 0);

    // Points
    svg.append("g")
      .selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.true))
      .attr("cy", d => y(d.debate))
      .attr("r", 4)
      .attr("fill", d => color(d.abs_error))
      .attr("opacity", 0.9)
      .on("mouseover", (event, d) => {{
        tooltip.transition().duration(150).style("opacity", .95);
        tooltip.html(`<strong>${{d.cell_line_name}}</strong><br/>Drug: ${{d.drug_name}}<br/>Actual: ${{d.true.toFixed(3)}}<br/>Pred: ${{d.debate.toFixed(3)}}<br/>Abs Error: ${{d.abs_error.toFixed(3)}}`)
          .style("left", (event.pageX + 12) + "px")
          .style("top", (event.pageY - 28) + "px");
      }})
      .on("mousemove", (event) => {{
        tooltip.style("left", (event.pageX + 12) + "px")
               .style("top", (event.pageY - 28) + "px");
      }})
      .on("mouseout", () => {{
        tooltip.transition().duration(200).style("opacity", 0);
      }});

    // Legend for absolute error
    const legendHeight = 120, legendWidth = 12;
    const legend = svg.append("g")
      .attr("transform", `translate(${{width - margin.right + 30}}, ${{margin.top + 20}})`);

    const legendScale = d3.scaleLinear()
      .domain([minErr, maxErr])
      .range([legendHeight, 0]);

    const legendAxis = d3.axisRight(legendScale)
      .ticks(5)
      .tickSize(6)
      .tickFormat(d3.format(".2f"));

    // Gradient
    const defs = svg.append("defs");
    const gradient = defs.append("linearGradient")
      .attr("id", "error-gradient")
      .attr("x1", "0%").attr("y1", "100%")
      .attr("x2", "0%").attr("y2", "0%");
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "#4f8ef7");
    gradient.append("stop").attr("offset", "50%").attr("stop-color", "#8e63ff");
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "#f16b6b");

    legend.append("rect")
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#error-gradient)");

    legend.append("g")
      .attr("transform", `translate(${{legendWidth}}, 0)`)
      .call(legendAxis)
      .selectAll("text")
      .style("fill", "#6d7890")
      .style("font-size", "11px");

    legend.append("text")
      .attr("x", -8)
      .attr("y", -10)
      .attr("fill", "#4a4f6b")
      .attr("font-size", "12px")
      .text("Absolute Error");
  </script>
</body>
</html>
"""

html_filled = (
    html_template
    .replace("__DATA_SPLIT__", data_split_html)
    .replace("__HYPERPARAM__", hyper_html)
    .replace("__RMSE__", metric_cards["rmse"])
    .replace("__PCC__", metric_cards["pcc"])
    .replace("__SCC__", metric_cards["scc"])
    .replace("__FOOT_RMSE__", footer_stats["rmse"])
    .replace("__FOOT_PCC__", footer_stats["pcc"])
    .replace("__FOOT_SCC__", footer_stats["scc"])
    .replace("__MERMAID_LINEAR__", mermaid_linear)
    .replace("__MERMAID_CELL__", mermaid_cell)
    .replace("__SUMMARY__", summary_html)
    .replace("__SCATTER__", json.dumps(scatter_data))
)

Path("{output_html}").write_text(html_filled, encoding="utf-8")
print(json.dumps({
    "status": "success",
    "html_path": "{output_html}",
    "records": len(df),
    "summary_from_llm": summary_llm,
    "summary_sources": {
        "plan": plan_text is not None,
        "final_report": report_text is not None,
        "agenda": agenda_text is not None
    },
    "mermaid": {
        "linear": linear_cfg is not None,
        "cell_encoder": cell_cfg is not None
    }
}))
''')

    script = (
        script
        .replace("{output_dir}", output_dir)
        .replace("{merge_csv}", merge_csv)
        .replace("{metrics_csv}", metrics_csv)
        .replace("{data_split_yaml}", data_split_yaml)
        .replace("{hyper_yaml}", hyper_yaml)
        .replace("{base_dir}", base_dir)
        .replace("{output_html}", output_html)
    )
    # Normalize escaped braces introduced for f-string safety.
    script = script.replace("{{", "{").replace("}}", "}")

    try:
        result = subprocess.run(
            ["docker", "exec", *env_flags, container_name, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=240
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            logger.error(f"❌ [HTML] Docker exec failed: {error_msg}")
            return {
                "messages": [AIMessage(content=f"HTML generation failed: {error_msg}")],
                "processing_logs": state.get("processing_logs", []) + [
                    f"[HTML] Error: {error_msg}"
                ]
            }

        output = result.stdout.strip()
        try:
            html_result = json.loads(output)
        except json.JSONDecodeError:
            html_result = {"status": "unknown", "raw_output": output}

        status = html_result.get("status", "unknown")
        html_path = html_result.get("html_path", "")
        records = html_result.get("records", 0)

        if status == "success":
            msg = f"✅ HTML report generated: {{html_path}} (records: {{records}})"
        else:
            msg = "HTML generation result: " + json.dumps(html_result, indent=2)

        logger.info(f"✅ [HTML] Status: {status} | File: {html_path}")

        return {
            "messages": [AIMessage(content=msg)],
            "processing_logs": state.get("processing_logs", []) + [
                f"[HTML] Status: {status}",
                f"[HTML] File: {html_path}"
            ]
        }

    except subprocess.TimeoutExpired:
        logger.error("❌ [HTML] Timeout after 240s")
        return {
            "messages": [AIMessage(content="HTML generation timed out after 240 seconds")],
            "processing_logs": state.get("processing_logs", []) + [
                "[HTML] Timeout"
            ]
        }
    except Exception as e:
        logger.error(f"❌ [HTML] Exception: {e}")
        return {
            "messages": [AIMessage(content=f"HTML generation error: {str(e)}")],
            "processing_logs": state.get("processing_logs", []) + [
                f"[HTML] Exception: {str(e)}"
            ]
        }


def _route_analysis(state: MARBLEState) -> str:
    """Route to visualization or unified html based on task_mode."""
    task = (state.get("task_mode") or "").lower()
    if task == "html":
        return "make_html"
    return "run_visualization"


def create_analysis_subgraph() -> Any:
    """Create analysis subgraph with visualization and mermaid generation."""
    logger.info("Building analysis subgraph (visualization + html)...")

    builder = StateGraph(MARBLEState)

    builder.add_node("analysis_router", lambda state: {})
    builder.add_node("run_visualization", _run_visualization)
    builder.add_node("make_html", _run_make_html)

    builder.set_entry_point("analysis_router")

    builder.add_conditional_edges(
        "analysis_router",
        _route_analysis,
        {
            "run_visualization": "run_visualization",
            "make_html": "make_html"
        }
    )

    builder.add_edge("run_visualization", END)
    builder.add_edge("make_html", END)

    subgraph = builder.compile()
    logger.info("✅ Analysis subgraph compiled (visualization + html)")
    return subgraph


def get_analysis_subgraph():
    """Factory function to create analysis subgraph."""
    global ANALYSIS_SUBGRAPH

    if ANALYSIS_SUBGRAPH is None:
        ANALYSIS_SUBGRAPH = create_analysis_subgraph()

    return ANALYSIS_SUBGRAPH
