"""Build Development Subgraph - Code Implementation Workflow.

Workflow:
1. Read Proposal: Determine which component to implement
2. Code Expert: Implement the architecture in the template
3. Validator: Check implementation correctness
4. Loop: If validation fails, Code Expert fixes until pass (max iterations)
5. Update Config: Update config.yaml with new component type

Input: Implementation proposal from build_debate_workflow
Output: Implemented vision_*.py files ready for use + updated config.yaml

Note: Workspace is already created by build_debate_workflow at experiments/build
"""

import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

from configs.config import LANGGRAPH_CONFIG
from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.utils import get_project_root

from .agents import CodeExpertAgent, ValidatorAgent

# Module-level cache
_BUILD_DEVELOPMENT_SUBGRAPH = None

# Constants
MAX_FIX_ITERATIONS = 10


# ==============================================================================
# PATH HELPERS - Use PROJECT_ROOT from environment
# ==============================================================================

# Model-specific path and component mapping (must match build_debate_workflow)
# This configuration allows dynamic addition of new bio models.
# To add a new model, simply add a new entry with all required fields.
MODEL_WORKFLOW_CONFIG = {
    "stagate": {
        # === Path Settings === (must match build_debate_workflow)
        "workspace": "experiments/build",  # Unified workspace for all models
        "components": ["encoder", "decoder"],
        "component_templates": {
            "encoder": "components/encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        # Config type values in config.yaml
        "config_type_mapping": {
            "encoder": "encoder_other",
            "decoder": "decoder_other",
        },
        # Expected class names in template files
        "expected_classes": {
            "encoder": "Encoder",
            "decoder": "Decoder",
        },
        # DataLoader classes to preserve (None if no DataLoader)
        "dataloader_info": {
            "encoder": None,
            "decoder": None,
        },
    },
    "deepst": {
        # === Path Settings ===
        "workspace": "experiments/build",
        "components": ["encoder", "decoder"],
        "component_templates": {
            "encoder": "components/encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        "config_type_mapping": {
            "encoder": "encoder_other",
            "decoder": "decoder_other",
        },
        "expected_classes": {
            "encoder": "Encoder",
            "decoder": "Decoder",
        },
        "dataloader_info": {
            "encoder": None,
            "decoder": None,
        },
    },
    "deeptta": {
        # === Path Settings ===
        "workspace": "experiments/build",
        "components": ["drug_encoder", "cell_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "cell_encoder": "components/cell_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        "config_type_mapping": {
            "drug_encoder": "drug_encoder_other",
            "cell_encoder": "cell_encoder_other",
            "decoder": "decoder_other",
        },
        "expected_classes": {
            "drug_encoder": "DrugEncoder",
            "cell_encoder": "CellEncoder",
            "decoder": "Decoder",
        },
        "dataloader_info": {
            "drug_encoder": "DrugDataLoader class",
            "cell_encoder": "CellDataLoader class",
            "decoder": None,
        },
    },
    "deepdr": {
        # === Path Settings ===
        "workspace": "experiments/build",
        "components": ["drug_encoder", "cell_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "cell_encoder": "components/cell_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        "config_type_mapping": {
            "drug_encoder": "drug_encoder_other",
            "cell_encoder": "cell_encoder_other",
            "decoder": "decoder_other",
        },
        "expected_classes": {
            "drug_encoder": "DrugEncoder",
            "cell_encoder": "CellEncoder",
            "decoder": "Decoder",
        },
        "dataloader_info": {
            "drug_encoder": "DrugDataLoader class",
            "cell_encoder": "CellDataLoader class",
            "decoder": None,
        },
    },
    "hyperattentiondti": {
        # === Path Settings ===
        "workspace": "experiments/build",
        "components": ["drug_encoder", "protein_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "protein_encoder": "components/protein_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        "config_type_mapping": {
            "drug_encoder": "drug_encoder_other",
            "protein_encoder": "protein_encoder_other",
            "decoder": "decoder_other",
        },
        "expected_classes": {
            "drug_encoder": "DrugEncoder",
            "protein_encoder": "ProteinEncoder",
            "decoder": "Decoder",
        },
        "dataloader_info": {
            "drug_encoder": None,
            "protein_encoder": None,
            "decoder": None,
        },
    },
    "dlm-dti": {
        # === Path Settings ===
        "workspace": "experiments/build",
        "components": ["drug_encoder", "protein_encoder", "decoder"],
        "component_templates": {
            "drug_encoder": "components/drug_encoder_other.py",
            "protein_encoder": "components/protein_encoder_other.py",
            "decoder": "components/decoder_other.py",
        },
        "config_type_mapping": {
            "drug_encoder": "drug_encoder_other",
            "protein_encoder": "protein_encoder_other",
            "decoder": "decoder_other",
        },
        "expected_classes": {
            "drug_encoder": "DrugEncoder",
            "protein_encoder": "ProteinEncoder",
            "decoder": "Decoder",
        },
        "dataloader_info": {
            "drug_encoder": "DrugDataLoader class",
            "protein_encoder": "ProteinDataLoader class",
            "decoder": None,
        },
    },
}

# Default fallback
DEFAULT_MODEL = "deeptta"

# Backward compatibility alias
MODEL_PATH_CONFIG = MODEL_WORKFLOW_CONFIG


def _get_model_config(model: str = None) -> dict:
    """Get full workflow configuration for a model."""
    if model and model in MODEL_WORKFLOW_CONFIG:
        return MODEL_WORKFLOW_CONFIG[model]
    return MODEL_WORKFLOW_CONFIG[DEFAULT_MODEL]


def _get_build_workspace_path(model: str = None, iteration: int = 1) -> Path:
    """Get build workspace path based on model and iteration.

    Each iteration has its own workspace: experiments/build_{iteration}/
    """
    project_root = get_project_root()
    return Path(project_root) / "experiments" / f"build_{iteration}"


def _get_debate_outputs_path(model: str = None, iteration: int = 1) -> Path:
    """Get build debate outputs path where proposal is stored."""
    return _get_build_workspace_path(model, iteration) / "build_debate_outputs"


def _get_template_path(component: str, model: str = None, iteration: int = 1) -> str:
    """Get template file path for component in the build workspace.

    Uses model-specific component_templates from config.
    """
    model_config = _get_model_config(model)
    build_workspace = _get_build_workspace_path(model, iteration)
    component_templates = model_config.get("component_templates", {})

    if component in component_templates:
        return str(build_workspace / component_templates[component])
    else:
        available = list(component_templates.keys())
        raise ValueError(f"Unknown component '{component}' for model '{model}'. Available: {available}")


# ==============================================================================
# DEBUG LOGGING HELPER
# ==============================================================================

def _save_debug_log(agent_name: str, iteration: int, messages: list, result_info: dict = None, model: str = None, current_iteration: int = 1):
    """Save agent messages to debug log file.

    Args:
        agent_name: Name of the agent (e.g., "code_expert", "validator")
        iteration: Current development iteration number (within build workflow)
        messages: List of messages from the agent
        result_info: Optional dict with additional result info
        model: Target model name for path resolution
        current_iteration: Global iteration number for build path
    """
    from datetime import datetime

    debug_dir = _get_debate_outputs_path(model, current_iteration) / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)

    log_file = debug_dir / f"development_debug.log"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {agent_name.upper()} - Iteration {iteration}\n")
        f.write("=" * 80 + "\n\n")

        for i, msg in enumerate(messages[-5:]):  # Last 5 messages
            msg_type = getattr(msg, 'type', type(msg).__name__)
            content = msg.content if hasattr(msg, 'content') else str(msg)

            f.write(f"--- Message {i+1} ({msg_type}) ---\n")
            if isinstance(content, str):
                f.write(content[:5000])  # Truncate long messages
                if len(content) > 5000:
                    f.write(f"\n... [truncated, total {len(content)} chars]")
            else:
                f.write(str(content)[:5000])
            f.write("\n\n")

        if result_info:
            f.write("--- Result Info ---\n")
            for key, value in result_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    logger.debug(f"[DEBUG_LOG] Saved {agent_name} iteration {iteration} to {log_file}")


# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================

def read_proposal_node(state: MARBLEState) -> Dict[str, Any]:
    """Read the proposal and determine which component to implement.

    This function is MODEL-AWARE: it detects components based on the target model.
    Supported models and their components:
    - deepst: encoder, decoder
    - stagate: encoder, decoder
    - deeptta: drug_encoder, cell_encoder, decoder
    - deepdr: drug_encoder, cell_encoder, decoder
    - dlm-dti: drug_encoder, protein_encoder, decoder
    - hyperattentiondti: drug_encoder, protein_encoder, decoder
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    valid_components = model_config.get("components", [])

    debate_outputs = _get_debate_outputs_path(target_model, current_iteration)
    proposal_path = debate_outputs / "implementation_proposal.md"

    if not proposal_path.exists():
        logger.error(f"[READ_PROPOSAL] Proposal not found: {proposal_path}")
        return {
            "processing_logs": ["[READ_PROPOSAL] ERROR: Proposal file not found"],
        }

    content = proposal_path.read_text(encoding='utf-8')
    content_lower = content.lower()

    logger.info(f"[READ_PROPOSAL] Model: {target_model}, Valid components: {valid_components}")

    # Extract component from proposal - MODEL-AWARE detection
    component = None

    # Build component pattern dynamically from valid_components
    # Create regex pattern from valid components
    component_pattern_str = "|".join([c.replace("_", "[ _]?") for c in valid_components])

    component_patterns = [
        # Explicit "Component to Modify" patterns
        (rf"component to modify[:\s]*\**\s*({component_pattern_str})", None),
        (rf"\*\*component to modify[:\*]*\s*\**\s*({component_pattern_str})", None),
        # Config type patterns (using _other suffix)
        *[(rf"{c}_other", c) for c in valid_components],
    ]

    # Add general mention patterns for each valid component
    for comp in valid_components:
        component_patterns.append((rf"\b{comp}\b", comp))

    # Default to first valid component
    default_component = valid_components[0] if valid_components else "encoder"

    # Try each pattern in order of priority
    for pattern, fixed_result in component_patterns:
        match = re.search(pattern, content_lower)
        if match:
            if fixed_result:
                component = fixed_result
            else:
                matched = match.group(1).strip().replace(" ", "_")
                # Normalize to exact component name
                for valid_comp in valid_components:
                    if valid_comp.replace("_", "") in matched.replace("_", "").replace(" ", ""):
                        component = valid_comp
                        break

            if component:
                logger.info(f"[READ_PROPOSAL] Found component via pattern '{pattern}': {component}")
                break

    # Validate component is valid for this model
    if component and component not in valid_components:
        logger.warning(f"[READ_PROPOSAL] Detected '{component}' but not valid for {target_model}. Valid: {valid_components}")
        component = None

    # Default if not found
    if not component:
        component = default_component
        logger.warning(f"[READ_PROPOSAL] Could not determine component, defaulting to {component}")

    logger.info(f"[READ_PROPOSAL] Target component: {component} (model: {target_model})")

    return {
        "target_component": component,
        "processing_logs": [f"[READ_PROPOSAL] Will implement {component} for {target_model}"],
    }


async def code_expert_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Code Expert implements the architecture.

    This function is MODEL-AWARE: uses dynamic configuration for class names and dataloader info.
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    debate_outputs = _get_debate_outputs_path(target_model, current_iteration)
    proposal_path = str(debate_outputs / "implementation_proposal.md")

    # Get default component from model config
    default_component = model_config["components"][0] if model_config["components"] else "encoder"
    target_component = state.get("target_component", default_component)
    template_path = _get_template_path(target_component, target_model, current_iteration)
    config_path = str(_get_build_workspace_path(target_model, current_iteration) / "config.yaml")

    iteration = state.get("development_iteration_count", 0)

    logger.info(f"[CODE_EXPERT] Iteration {iteration + 1}: Implementing {target_component} for {target_model}")

    agent = CodeExpertAgent(
        proposal_path=proposal_path,
        target_component=target_component,
        template_path=template_path,
        config_path=config_path,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    # Get model-specific configuration dynamically
    expected_class = model_config.get("expected_classes", {}).get(target_component, "Encoder")
    dataloader_info = model_config.get("dataloader_info", {}).get(target_component)

    # Build dataloader preservation instruction
    if dataloader_info:
        dataloader_instruction = f"""## SUPER CRITICAL: PRESERVE DATALOADER CLASSES
You MUST preserve the {dataloader_info} EXACTLY as they are in the template.
Copy them without ANY modifications when writing the file."""
    else:
        dataloader_instruction = "## Note: No DataLoader classes to preserve for this component."

    if iteration == 0:
        # First iteration: full implementation
        prompt = f"""Implement the {target_component} architecture.

## CRITICAL: Target Information
- Model: {target_model}
- Component: {target_component}
- Expected Class Name: {expected_class}
- File to modify: {template_path}
- Config file: {config_path}
- Proposal: {proposal_path}

## Steps
1. Read the proposal: {proposal_path}
2. Find the section "## 3. Code to Implement" in the proposal
3. Read the ENTIRE template: {template_path}
4. Write the COMPLETE file using write_file (NOT replace_in_file)
5. **UPDATE CONFIG**: Apply config changes from proposal using update_yaml_config

## SUPER CRITICAL: COPY CODE FROM PROPOSAL EXACTLY
The proposal contains WORKING CODE in "## 3. Code to Implement" section.
You MUST:
1. Read the proposal and find ALL class definitions in "## 3. Code to Implement"
2. Identify the MAIN class (nn.Module with forward method) and AUXILIARY classes
3. Copy ALL auxiliary classes into the AUXILIARY MODULES section of the template
4. ADAPT the main class code for the {expected_class} class:
   - Keep the class name as {expected_class}
   - Copy the __init__ logic from the proposal's main class
   - Copy the forward method logic from the proposal's main class
   - Adapt parameter names to use config dict pattern

## DO NOT:
- Write your own implementation from scratch
- Use a DIFFERENT architecture than what's in the proposal
- Skip ANY auxiliary classes mentioned in the proposal
- Replace the proposal's architecture with something else (e.g., if proposal uses LSTM, don't use Transformer)

## Key Requirements
- Keep main class name as {expected_class}
- Remove the `raise NotImplementedError` lines
- Implement __init__, get_output_dim, and forward methods
- Read config parameters BEFORE using them
- Use `device = next(self.parameters()).device`

{dataloader_instruction}

## MANDATORY: Update Config Parameters

After writing the code, you MUST update {config_path} with proposed parameter changes.
Look for "## 6. Config Changes" section in the proposal and apply ALL changes.

Example:
```
update_yaml_config("{config_path}", "model.encoder.architecture.hidden_dim", "256")
update_yaml_config("{config_path}", "model.encoder.architecture.heads", "4")
update_yaml_config("{config_path}", "model.encoder.architecture.dropout", "0.2")
update_yaml_config("{config_path}", "model.decoder.architecture.hidden_dim", "256")
update_yaml_config("{config_path}", "training.learning_rate", "0.0005")
```

Use write_file to write the COMPLETE file at once, then use update_yaml_config for each config change.
"""
    else:
        # Fix iteration: use validator feedback
        validator_feedback = state.get("validator_feedback", "No specific feedback")

        # Build fix dataloader instruction
        if dataloader_info:
            fix_dataloader = f"## CRITICAL: Preserve DataLoader Classes\nWhen rewriting, PRESERVE the {dataloader_info} exactly."
        else:
            fix_dataloader = ""

        prompt = f"""Fix the implementation based on validator feedback.

## Validator Feedback
{validator_feedback}

## Target File
{template_path}

## IMPORTANT: You Can Disagree with Validator
The validator may be wrong. Before fixing:
1. Read the current code using read_file
2. Evaluate if each issue is REAL or a FALSE POSITIVE
3. If validator is wrong, state "REBUTTAL: [explanation]"
4. Only fix REAL issues

## If Fixing
Use write_file to write the COMPLETE fixed file.
DO NOT use replace_in_file (causes duplication).

{fix_dataloader}
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Save debug log
    _save_debug_log(
        agent_name="code_expert",
        iteration=iteration + 1,
        messages=result.get("messages", []),
        result_info={"target_component": target_component, "template_path": template_path},
        model=target_model
    )

    # Extract REBUTTAL from code expert's response if present
    code_expert_rebuttal = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) or getattr(msg, 'type', '') == 'ai':
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "REBUTTAL:" in content.upper():
                # Extract the rebuttal section
                rebuttal_idx = content.upper().find("REBUTTAL:")
                code_expert_rebuttal = content[rebuttal_idx:]
                # Take up to 2000 chars to avoid token overflow
                code_expert_rebuttal = code_expert_rebuttal[:2000]
                logger.info(f"[CODE_EXPERT] REBUTTAL detected: {code_expert_rebuttal[:100]}...")
                break

    return {
        "messages": result["messages"],
        "development_iteration_count": iteration + 1,
        "code_expert_rebuttal": code_expert_rebuttal,
        "processing_logs": [f"[CODE_EXPERT] Iteration {iteration + 1} completed"],
    }


async def validator_node(state: MARBLEState, checkpointer) -> Dict[str, Any]:
    """Validator checks the implementation.

    This function is MODEL-AWARE: uses dynamic configuration for expected class names and dataloader checks.
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    debate_outputs = _get_debate_outputs_path(target_model, current_iteration)
    proposal_path = str(debate_outputs / "implementation_proposal.md")

    # Get default component from model config
    default_component = model_config["components"][0] if model_config["components"] else "encoder"
    target_component = state.get("target_component", default_component)
    code_path = _get_template_path(target_component, target_model, current_iteration)

    logger.info(f"[VALIDATOR] Checking {target_component} implementation for {target_model}")

    agent = ValidatorAgent(
        proposal_path=proposal_path,
        code_path=code_path,
        target_component=target_component,
        checkpointer=checkpointer
    )
    agent.initialize_agent()

    # Get model-specific configuration dynamically
    expected_class = model_config.get("expected_classes", {}).get(target_component, "Encoder")
    dataloader_info = model_config.get("dataloader_info", {}).get(target_component)

    # Build dataloader check instruction
    if dataloader_info:
        dataloader_check = f"Check that {dataloader_info} exist and are complete"
    else:
        dataloader_check = "No DataLoader classes to check for this component."

    # Check if code expert provided a rebuttal
    code_expert_rebuttal = state.get("code_expert_rebuttal", "")
    rebuttal_section = ""
    if code_expert_rebuttal:
        rebuttal_section = f"""
## ⚠️ CODE EXPERT REBUTTAL (MUST CONSIDER!)

The Code Expert has provided a rebuttal to your previous validation feedback.
**You MUST read and seriously consider this rebuttal before making your decision.**

### Code Expert's Rebuttal:
{code_expert_rebuttal}

### How to Handle the Rebuttal:
1. **Read the rebuttal carefully** - understand what the Code Expert is arguing
2. **Verify their claims** - check if the code actually does what they claim
3. **If the rebuttal is VALID**:
   - Acknowledge that your previous feedback was incorrect
   - Change your validation to PASS if there are no other real issues
   - State: "REBUTTAL ACCEPTED: [explanation]"
4. **If the rebuttal is INVALID**:
   - Explain WHY it's invalid with specific evidence
   - Point to the exact code that contradicts their claim
   - State: "REBUTTAL REJECTED: [explanation with evidence]"

**DO NOT simply ignore the rebuttal or repeat your previous feedback without addressing it.**
"""
        logger.info(f"[VALIDATOR] Including code expert rebuttal in prompt")

    prompt = f"""Validate the {target_component} implementation.
{rebuttal_section}

## Target Information
- Model: {target_model}
- Component: {target_component}
- Expected Class: {expected_class}
- Code File: {code_path}
- Proposal: {proposal_path}

## Validation Steps (IN ORDER)

### Step 1: SYNTAX CHECK (MANDATORY FIRST)
Use check_python_syntax tool on {code_path}
- If FAIL → Immediately return VALIDATION: FAIL

### Step 2: Import Check
Use check_imports tool on {code_path}

### Step 3: Structure Check
- Class name is {expected_class}
- Has __init__, get_output_dim, forward methods (ONE each, no duplicates)

### Step 4: DataLoader Preservation Check
{dataloader_check}

### Step 5: PROPOSAL CONSISTENCY CHECK (CRITICAL!)
Read the proposal file and compare with the implemented code:
1. Find what classes are defined in "## 3. Code to Implement" section of proposal
2. Check if AUXILIARY MODULES section in code file contains those auxiliary classes
3. Check if the architecture in code MATCHES the proposal (same layer types, same structure)

FAIL if there is ARCHITECTURE MISMATCH:
- Proposal defines auxiliary classes → Code is missing them in AUXILIARY MODULES section
- Proposal uses specific layer type (e.g., LSTM, GNN, Attention) → Code uses different type
- Proposal has specific structure → Code has completely different structure

### Step 6: Logic Check
- Architecture matches proposal implementation

## IMPORTANT: Be Careful Not to Hallucinate
- Only report issues you are 100% certain about
- Mark uncertain issues as "POTENTIAL ISSUE"
- Don't invent problems that don't exist

## Response Format
End with either:
- VALIDATION: PASS (if all checks pass)
- VALIDATION: FAIL (only for 100% certain issues)

## CRITICAL: If FAIL, provide ACTIONABLE feedback

For EACH issue, you MUST provide:
1. **Location**: Method name + line number (e.g., "In `forward()`, line 45")
2. **Problem**: What is wrong (e.g., "Variable `device` used but not defined")
3. **Hint**: Direction to fix WITHOUT complete code (e.g., "Use `next(self.parameters()).device` to get device")

Example:
```
Issue 1:
- Location: AUXILIARY MODULES section
- Problem: Missing auxiliary class from proposal
- Hint: Copy the auxiliary class from proposal's "## 3. Code to Implement" section

Issue 2:
- Location: `__init__()` method
- Problem: Architecture mismatch - code uses different layer types than proposal
- Hint: Use the same layer types as defined in proposal's "## 3. Code to Implement" section
```

DO NOT provide complete corrected code. Only give hints and direction.
"""

    result = await agent.compiled_agent.ainvoke({
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=prompt)]
    })

    # Check validation result from messages
    validation_passed = False
    validator_feedback = ""

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) or getattr(msg, 'type', '') == 'ai':
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Normalize: remove extra whitespace/newlines for robust matching
            content_normalized = ' '.join(content.upper().split())

            # Method 1: Look for "VALIDATION: PASS" or "VALIDATION: FAIL" pattern
            # (handles cases where colon might be on different line)
            if "VALIDATION" in content_normalized or "VERDICT" in content_normalized:
                # Check for PASS pattern (various formats)
                if any(pattern in content_normalized for pattern in [
                    "VALIDATION: PASS",
                    "VALIDATION : PASS",
                    "VALIDATION:PASS",
                    "FINAL VERDICT VALIDATION : PASS",
                    "FINAL VERDICT : PASS",
                    "VERDICT : PASS",
                    "VERDICT: PASS",
                ]):
                    validation_passed = True
                    logger.info("[VALIDATOR] Detected PASS via normalized pattern")
                # Check for FAIL pattern
                elif any(pattern in content_normalized for pattern in [
                    "VALIDATION: FAIL",
                    "VALIDATION : FAIL",
                    "VALIDATION:FAIL",
                    "FINAL VERDICT VALIDATION : FAIL",
                    "FINAL VERDICT : FAIL",
                    "VERDICT : FAIL",
                    "VERDICT: FAIL",
                ]):
                    validation_passed = False
                    validator_feedback = content
                    logger.info("[VALIDATOR] Detected FAIL via normalized pattern")
                # Fallback: just check if PASS or FAIL appears near VALIDATION/VERDICT
                else:
                    # Find position of VALIDATION or VERDICT and check nearby words
                    val_pos = content_normalized.find("VALIDATION")
                    if val_pos == -1:
                        val_pos = content_normalized.find("VERDICT")
                    # Check within 50 chars after the keyword
                    nearby = content_normalized[val_pos:val_pos+50]
                    if "PASS" in nearby and "FAIL" not in nearby:
                        validation_passed = True
                        logger.info("[VALIDATOR] Detected PASS via proximity check")
                    elif "FAIL" in nearby:
                        validation_passed = False
                        validator_feedback = content
                        logger.info("[VALIDATOR] Detected FAIL via proximity check")
                break

    logger.info(f"[VALIDATOR] Result: {'PASS' if validation_passed else 'FAIL'}")

    # Save debug log
    iteration = state.get("development_iteration_count", 0)
    _save_debug_log(
        agent_name="validator",
        iteration=iteration,
        messages=result.get("messages", []),
        result_info={
            "validation_passed": validation_passed,
            "target_component": target_component,
            "code_path": code_path
        },
        model=target_model
    )

    return {
        "messages": result["messages"],
        "validation_passed": validation_passed,
        "validator_feedback": validator_feedback,
        "processing_logs": [f"[VALIDATOR] {'PASS' if validation_passed else 'FAIL'}"],
    }


def _extract_config_changes_from_file(file_path: Path) -> Dict[str, Any]:
    """Extract config change suggestions from a single proposal file.

    Looks for patterns like:
    - Parameter: encoder.architecture.hidden_dim
      - Current → Proposed: 10 → 256

    Or markdown format:
    - **Parameter**: encoder.architecture.hidden_dim
    - **Current → Proposed**: 10 → 512

    Returns dict like: {"encoder.architecture.hidden_dim": "256", ...}
    """
    if not file_path.exists():
        return {}

    content = file_path.read_text(encoding='utf-8')
    config_changes = {}

    # Pattern 1: "- Parameter: path.to.param" (plain or with **)
    # Matches both "- Parameter:" and "- **Parameter**:"
    param_pattern = r'-\s*\*?\*?Parameter\*?\*?:\s*([a-zA-Z0-9_.]+)'

    # Pattern 2: "Current → Proposed: old → new" or just values with arrows
    # Matches both "- Current → Proposed:" and "- **Current → Proposed**:"
    value_pattern = r'(?:Current\s*→\s*Proposed|Proposed)[:\s]*(?:\*?\*?)?[:\s]*(?:[^→\n]+→\s*)?(\d+(?:\.\d+)?(?:e-?\d+)?)'

    # Find all parameter mentions
    lines = content.split('\n')
    current_param = None

    for i, line in enumerate(lines):
        # Look for parameter
        param_match = re.search(param_pattern, line, re.IGNORECASE)
        if param_match:
            current_param = param_match.group(1).strip()
            continue

        # Look for value if we have a parameter
        if current_param:
            value_match = re.search(value_pattern, line, re.IGNORECASE)
            if value_match:
                value = value_match.group(1).strip()
                # Clean up the value (remove trailing punctuation, etc.)
                value = re.sub(r'[,\s]+$', '', value)
                if value:  # Only add if we got a value
                    config_changes[current_param] = value
                    logger.info(f"[EXTRACT_CONFIG] Found in {file_path.name}: {current_param} → {value}")
                current_param = None

    # Pattern 3: Direct "param: old → new" format (common in proposals)
    # e.g., "encoder.architecture.hidden_dim: 10 → 256"
    direct_pattern = r'((?:model\.)?(?:encoder|decoder|training)\.(?:architecture\.)?[a-zA-Z0-9_]+)[:\s]+(?:\d+(?:\.\d+)?(?:e-?\d+)?)\s*→\s*(\d+(?:\.\d+)?(?:e-?\d+)?)'
    for match in re.finditer(direct_pattern, content, re.IGNORECASE):
        param = match.group(1).strip()
        value = match.group(2).strip()
        if param not in config_changes:
            config_changes[param] = value
            logger.info(f"[EXTRACT_CONFIG] Found (direct) in {file_path.name}: {param} → {value}")

    # Pattern 4: Look in "## 6. Config Changes" section specifically
    config_section_match = re.search(r'##\s*\d*\.?\s*Config Changes.*?(?=##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if config_section_match:
        config_section = config_section_match.group(0)

        # Within this section, find all param → value pairs
        # Format: "- Parameter: encoder.architecture.hidden_dim" followed by "Current → Proposed: 10 → 256"
        section_lines = config_section.split('\n')
        current_param = None

        for line in section_lines:
            # Look for parameter line
            param_match = re.search(r'-\s*\*?\*?Parameter\*?\*?:\s*([a-zA-Z0-9_.]+)', line, re.IGNORECASE)
            if param_match:
                current_param = param_match.group(1).strip()
                continue

            # Look for value line
            if current_param:
                # Match "Current → Proposed: 10 → 256" or just "10 → 256"
                value_match = re.search(r'→\s*(\d+(?:\.\d+)?(?:e-?\d+)?)\s*(?:\(|$|\n)', line)
                if value_match:
                    value = value_match.group(1).strip()
                    if value and current_param not in config_changes:
                        config_changes[current_param] = value
                        logger.info(f"[EXTRACT_CONFIG] Found (section) in {file_path.name}: {current_param} → {value}")
                    current_param = None

    return config_changes


def _extract_config_changes_from_proposal(proposal_path: Path) -> Dict[str, Any]:
    """Extract config change suggestions from multiple proposal files.

    Checks in order of priority:
    1. implementation_proposal.md (primary)
    2. proposal_round*_article*.md (debate proposals with detailed config changes)
    3. final_rank1_proposal.md (final selection)

    Returns dict like: {"encoder.architecture.hidden_dim": "512", ...}
    """
    config_changes = {}
    debate_outputs = proposal_path.parent

    # Priority 1: implementation_proposal.md
    if proposal_path.exists():
        changes = _extract_config_changes_from_file(proposal_path)
        config_changes.update(changes)
        if changes:
            logger.info(f"[EXTRACT_CONFIG] Found {len(changes)} changes in implementation_proposal.md")

    # Priority 2: proposal_round*_article*.md (these often have detailed Config Changes sections)
    # Only check if we haven't found config changes yet
    if not config_changes:
        import glob
        proposal_files = sorted(glob.glob(str(debate_outputs / "proposal_round*_article*.md")))
        for pf in proposal_files:
            changes = _extract_config_changes_from_file(Path(pf))
            if changes:
                config_changes.update(changes)
                logger.info(f"[EXTRACT_CONFIG] Found {len(changes)} changes in {Path(pf).name}")

    # Priority 3: response_round*_article*.md (responses may also have config suggestions)
    if not config_changes:
        response_files = sorted(glob.glob(str(debate_outputs / "response_round*_article*.md")))
        for rf in response_files:
            changes = _extract_config_changes_from_file(Path(rf))
            if changes:
                config_changes.update(changes)
                logger.info(f"[EXTRACT_CONFIG] Found {len(changes)} changes in {Path(rf).name}")

    if not config_changes:
        logger.warning("[EXTRACT_CONFIG] No config changes found in any proposal files")

    return config_changes


def _update_yaml_value(lines: list, param_path: str, new_value: str) -> Tuple[bool, str]:
    """Update a specific parameter in YAML lines.

    param_path: e.g., "encoder.architecture.hidden_dim"
    Returns: (success, old_value)
    """
    parts = param_path.split('.')
    if len(parts) < 2:
        return False, None

    # Find the nested parameter
    current_depth = 0
    target_depth = len(parts) - 1
    found_sections = []
    expected_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue

        current_indent = len(line) - len(stripped)

        # Check if this line starts a section we're looking for
        for depth, part in enumerate(parts[:-1]):
            if depth == len(found_sections) and stripped.startswith(f"{part}:"):
                found_sections.append((i, current_indent))
                expected_indent = current_indent + 2  # Expect children to be indented
                break

        # If we've found all parent sections, look for the final key
        if len(found_sections) == target_depth:
            final_key = parts[-1]
            if stripped.startswith(f"{final_key}:") and current_indent >= expected_indent:
                # Extract old value
                old_value = stripped.split(":", 1)[1].strip()
                # Remove inline comment for comparison
                old_value_clean = old_value.split("#")[0].strip()

                # Preserve indentation and comment
                indent = line[:current_indent]
                comment_part = ""
                if "#" in stripped:
                    comment_idx = stripped.index("#")
                    comment_part = "  " + stripped[comment_idx:]

                # Update line
                lines[i] = f"{indent}{final_key}: {new_value}{comment_part}\n"
                return True, old_value_clean

    return False, None


# Parameters that CAN be modified (from "# Can touch" sections)
ALLOWED_CONFIG_PARAMS = {
    # encoder section
    "encoder.architecture.hidden_dim",
    "encoder.architecture.heads",
    "encoder.architecture.dropout",
    "encoder.type",
    # decoder section
    "decoder.architecture.hidden_dim",
    "decoder.architecture.heads",
    "decoder.architecture.dropout",
    "decoder.type",
    # training section
    "training.epochs",
    "training.learning_rate",
    "training.weight_decay",
    "training.gradient_clipping",
}


def update_config_node(state: MARBLEState) -> Dict[str, Any]:
    """Update config.yaml with the new component type AND hyperparameter changes.

    1. Updates the component type (from proposal target)
    2. Applies hyperparameter changes from implementation_proposal.md
       - ONLY updates "# Can touch" sections (encoder, decoder, training)
       - NEVER touches "# Never touch" sections (data, clustering, evaluation, output, logging)

    Uses line-by-line replacement to preserve formatting and comments.
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    # Use first available component as default (model-specific)
    default_component = model_config["components"][0] if model_config.get("components") else "encoder"
    target_component = state.get("target_component", default_component)
    workspace_path = _get_build_workspace_path(target_model, current_iteration)
    config_path = workspace_path / "config.yaml"
    proposal_path = _get_debate_outputs_path(target_model, current_iteration) / "implementation_proposal.md"

    if not config_path.exists():
        logger.error(f"[UPDATE_CONFIG] Config not found: {config_path}")
        return {
            "processing_logs": ["[UPDATE_CONFIG] ERROR: config.yaml not found"],
        }

    # Read config file line by line
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    update_logs = []

    # =========================================================================
    # STEP 1: Update component type
    # =========================================================================
    type_mapping = model_config.get("config_type_mapping", {
        "encoder": "encoder_other",
        "decoder": "decoder_other",
        "drug_encoder": "drug_encoder_other",
        "cell_encoder": "cell_encoder_other",
    })

    if target_component in type_mapping:
        new_type = type_mapping[target_component]

        # Find and replace the type line for the target component
        in_target_section = False
        section_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            if stripped.startswith(f"{target_component}:"):
                in_target_section = True
                section_indent = len(line) - len(stripped)
                continue

            if in_target_section:
                current_indent = len(line) - len(stripped)

                if stripped and current_indent <= section_indent and not stripped.startswith('#'):
                    in_target_section = False
                    continue

                if stripped.startswith("type:"):
                    old_type = stripped.split(":", 1)[1].strip().split("#")[0].strip()
                    indent = line[:len(line) - len(stripped)]
                    comment_part = ""
                    if "#" in stripped:
                        comment_part = "  " + stripped[stripped.index("#"):]

                    lines[i] = f"{indent}type: {new_type}{comment_part}\n"
                    in_target_section = False
                    update_logs.append(f"{target_component}.type: {old_type} → {new_type}")
                    logger.info(f"[UPDATE_CONFIG] {target_component}.type: {old_type} → {new_type}")

    # =========================================================================
    # STEP 2: Apply hyperparameter changes from proposal (ONLY "# Can touch")
    # =========================================================================
    config_changes = _extract_config_changes_from_proposal(proposal_path)

    for param_path, new_value in config_changes.items():
        # Normalize param_path: remove "model." prefix if present
        normalized_path = param_path
        if param_path.startswith("model."):
            normalized_path = param_path[6:]  # Remove "model." prefix

        # Security check: only allow modifying "# Can touch" parameters
        if normalized_path not in ALLOWED_CONFIG_PARAMS:
            logger.warning(f"[UPDATE_CONFIG] Skipping disallowed param: {param_path} (not in Can touch)")
            continue

        # Clean up value: remove trailing annotations like "(no change)"
        clean_value = new_value.split("(")[0].strip()

        # Skip if value is not a valid number/value
        if not clean_value or clean_value == "":
            logger.warning(f"[UPDATE_CONFIG] Skipping empty value for: {param_path}")
            continue

        success, old_value = _update_yaml_value(lines, normalized_path, clean_value)
        if success:
            update_logs.append(f"{normalized_path}: {old_value} → {clean_value}")
            logger.info(f"[UPDATE_CONFIG] {normalized_path}: {old_value} → {clean_value}")
        else:
            logger.warning(f"[UPDATE_CONFIG] Could not find param: {normalized_path}")

    # =========================================================================
    # STEP 3: Write updated config
    # =========================================================================
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    if update_logs:
        logger.info(f"[UPDATE_CONFIG] Config updated successfully: {len(update_logs)} changes")
    else:
        logger.warning("[UPDATE_CONFIG] No changes applied to config")

    return {
        "processing_logs": [f"[UPDATE_CONFIG] Applied {len(update_logs)} changes: {'; '.join(update_logs)}"],
    }


def verify_config_node(state: MARBLEState) -> Dict[str, Any]:
    """Verify that config.yaml was updated correctly using Python code.

    This is a pure Python verification - no LLM involved.
    Checks that the target component type matches the expected vision type.
    """
    target_model = state.get("target_model")
    current_iteration = state.get("current_iteration", 1)
    model_config = _get_model_config(target_model)
    # Use first available component as default (model-specific)
    default_component = model_config["components"][0] if model_config.get("components") else "encoder"
    target_component = state.get("target_component", default_component)
    config_path = _get_build_workspace_path(target_model, current_iteration) / "config.yaml"

    # Get model-specific type mapping
    type_mapping = model_config.get("config_type_mapping", {
        "encoder": "encoder_other",
        "decoder": "decoder_other",
        "drug_encoder": "drug_encoder_other",
        "cell_encoder": "cell_encoder_other",
    })

    expected_type = type_mapping.get(target_component)
    if not expected_type:
        return {
            "config_verified": False,
            "processing_logs": [f"[VERIFY_CONFIG] FAIL: Unknown component {target_component}"],
        }

    # Read and parse config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"[VERIFY_CONFIG] Failed to read config: {e}")
        return {
            "config_verified": False,
            "processing_logs": [f"[VERIFY_CONFIG] FAIL: Could not read config - {e}"],
        }

    # Check the component type
    actual_type = config.get('model', {}).get(target_component, {}).get('type', '')

    if actual_type == expected_type:
        logger.info(f"[VERIFY_CONFIG] PASS: {target_component}.type = {actual_type}")
        return {
            "config_verified": True,
            "processing_logs": [f"[VERIFY_CONFIG] PASS: {target_component}.type = {actual_type}"],
        }
    else:
        logger.error(f"[VERIFY_CONFIG] FAIL: {target_component}.type = {actual_type}, expected {expected_type}")
        return {
            "config_verified": False,
            "processing_logs": [f"[VERIFY_CONFIG] FAIL: {target_component}.type = {actual_type}, expected {expected_type}"],
        }


# ==============================================================================
# ROUTING
# ==============================================================================

def route_after_validation(state: MARBLEState) -> str:
    """Route after validation check.

    Checks validation result from processing_logs first (most reliable),
    then falls back to validation_passed state variable.

    Routes:
    - PASS → update_config (then END)
    - FAIL → code_expert (for fixes)
    - Max iterations reached → update_config (best effort)
    """
    iteration = state.get("development_iteration_count", 0)

    # Method 1: Check processing_logs for "[VALIDATOR] PASS" (most reliable)
    processing_logs = state.get("processing_logs", [])
    for log in reversed(processing_logs):  # Check most recent logs first
        if "[VALIDATOR] PASS" in log:
            logger.info("[ROUTE] Validation passed (from processing_logs) → update_config")
            return "update_config"
        elif "[VALIDATOR] FAIL" in log:
            if iteration >= MAX_FIX_ITERATIONS:
                logger.warning(f"[ROUTE] Max iterations ({MAX_FIX_ITERATIONS}) reached → update_config (best effort)")
                return "update_config"
            logger.info(f"[ROUTE] Validation failed (from processing_logs), iteration {iteration} → code_expert")
            return "code_expert"

    # Method 2: Fallback to validation_passed state variable
    validation_passed = state.get("validation_passed", False)

    if validation_passed:
        logger.info("[ROUTE] Validation passed (from state) → update_config")
        return "update_config"

    if iteration >= MAX_FIX_ITERATIONS:
        logger.warning(f"[ROUTE] Max iterations ({MAX_FIX_ITERATIONS}) reached → update_config (best effort)")
        return "update_config"

    logger.info(f"[ROUTE] Validation failed (from state), iteration {iteration} → code_expert")
    return "code_expert"


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

def create_build_development_subgraph() -> Tuple[Any, Any]:
    """Create the build development subgraph.

    Workflow:
    1. read_proposal → code_expert → validator
    2. validator → (PASS) → update_config → verify_config → END
                 → (FAIL) → code_expert (loop)
    """
    builder = StateGraph(MARBLEState)
    checkpointer = InMemorySaver()

    # Add nodes
    builder.add_node("read_proposal", read_proposal_node)
    builder.add_node("code_expert", partial(code_expert_node, checkpointer=checkpointer))
    builder.add_node("validator", partial(validator_node, checkpointer=checkpointer))
    builder.add_node("update_config", update_config_node)
    builder.add_node("verify_config", verify_config_node)

    # Define edges
    builder.set_entry_point("read_proposal")
    builder.add_edge("read_proposal", "code_expert")
    builder.add_edge("code_expert", "validator")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "code_expert": "code_expert",
            "update_config": "update_config",
        }
    )

    # After config update, verify it
    builder.add_edge("update_config", "verify_config")

    # After verification, end the workflow
    builder.add_edge("verify_config", END)

    # Compile
    subgraph = builder.compile(checkpointer=checkpointer).with_config(
        recursion_limit=LANGGRAPH_CONFIG["recursion_limit"]
    )

    logger.info("[BUILD_DEVELOPMENT] Subgraph created with config update & verify")
    return subgraph, checkpointer


def get_build_development_subgraph():
    """Factory function returning compiled subgraph."""
    global _BUILD_DEVELOPMENT_SUBGRAPH
    if _BUILD_DEVELOPMENT_SUBGRAPH is None:
        _BUILD_DEVELOPMENT_SUBGRAPH = create_build_development_subgraph()
    return _BUILD_DEVELOPMENT_SUBGRAPH
