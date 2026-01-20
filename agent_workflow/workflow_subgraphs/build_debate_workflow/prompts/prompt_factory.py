"""Dynamic Prompt Factory for Build Debate Workflow.

This module provides dynamic prompt generation based on model configuration.
It allows adding new bioinformatics models without modifying prompt code.

Usage:
    from .prompt_factory import PromptFactory

    factory = PromptFactory(model_name="stagate")
    prompt = factory.get_weakness_analysis_prompt(
        target_summary_path="/path/to/summary.md",
        output_path="/path/to/output.md"
    )
"""

from typing import Dict, List, Optional, Any


class PromptFactory:
    """Factory class for generating model-specific prompts dynamically.

    This class takes model configuration and generates appropriate prompts
    for each stage of the build debate workflow.
    """

    def __init__(self, model_config: Dict[str, Any], iteration_context: str = ""):
        """Initialize PromptFactory with model configuration.

        Args:
            model_config: Model configuration dict from MODEL_WORKFLOW_CONFIG
                Required keys:
                - model_name: Display name (e.g., "STAGATE", "DeepTTA")
                - domain: Domain description (e.g., "spatial transcriptomics clustering")
                - components: List of component names (e.g., ["encoder", "decoder"])
                Optional keys:
                - domain_description: Longer description of the domain
            iteration_context: 이전 iteration의 학습 내용 (EvolvingMemory에서 생성)
        """
        self.config = model_config
        self.model_name = model_config.get("model_name", "Unknown")
        self.domain = model_config.get("domain", "machine learning")
        self.domain_description = model_config.get("domain_description", "")
        self.components = model_config.get("components", ["encoder", "decoder"])
        self.iteration_context = iteration_context

    @property
    def iteration_context_section(self) -> str:
        """이전 iteration 컨텍스트 섹션 생성 (파일 경로만 제공)."""
        if not self.iteration_context:
            return ""
        return f"""
## Previous Iteration Reference
Read the memory file for previous iteration results: {self.iteration_context}
- Use read_file tool to read this JSON file
- Check previous metrics, lessons learned, and failed approaches
- Avoid repeating failed approaches
"""

    @property
    def components_display(self) -> str:
        """Get component display string for prompts."""
        return " / ".join(self.components) + " / overall"

    @property
    def components_choice(self) -> str:
        """Get component choice string (without 'overall')."""
        return " / ".join(self.components)

    # =========================================================================
    # PAPER READER PROMPTS
    # =========================================================================

    def get_paper_reader_prompt(
        self,
        pdf_path: str,
        output_path: str,
    ) -> str:
        """Generate prompt for reading the target model paper."""
        # Build component sections dynamically
        component_sections = []
        for comp in self.components:
            comp_display = comp.replace("_", " ").title()
            component_sections.append(f"""### {comp_display}
- Input: [what input does it take]
- Architecture: [detailed architecture description]
- Output: [output dimension and meaning]""")

        component_text = "\n\n".join(component_sections)

        return f"""You are a {self.model_name} Paper Expert Agent.

## Your Task
Read the {self.model_name} paper PDF and create a comprehensive summary focusing on the METHOD section.

## Domain Context
- Model: {self.model_name}
- Domain: {self.domain}
- Description: {self.domain_description}

## PDF Location
{pdf_path}

## Instructions
1. Use the read_pdf tool to read the paper
2. Focus on extracting:
   - Model architecture overview
   - Each component's design and function
   - Training methodology
   - Key innovations and contributions
   - Any limitations mentioned

3. Write a structured markdown summary

## Output Format
Write a markdown file with the following structure:

```markdown
# {self.model_name} Paper Summary

## Overview
[Brief description of what {self.model_name} does for {self.domain}]

## Architecture

{component_text}

## Key Innovations
[List the main contributions]

## Limitations
[Any limitations mentioned or observable]
```

Save the summary to: {output_path}
"""

    # =========================================================================
    # WEAKNESS ANALYSIS PROMPT
    # =========================================================================

    def get_weakness_analysis_prompt(
        self,
        target_summary_path: str,
        output_path: str,
        config_path: str = None,
    ) -> str:
        """Generate prompt for analyzing target model weaknesses."""
        config_section = ""
        if config_path:
            config_section = f"""
## Config File Analysis
- Config Path: {config_path}

You MUST also analyze the config file and identify potential hyperparameter issues.
The config file has two types of sections:
- `# Can touch`: Parameters that CAN be modified (encoder, decoder, training settings)
- `# Never touch`: Parameters that MUST NOT be modified (data paths, clustering, evaluation, output)

Focus your config analysis ONLY on `# Can touch` sections:
- encoder: architecture settings (hidden_dim, heads, dropout)
- decoder: architecture settings (hidden_dim, heads, dropout)
- training: epochs, learning_rate, weight_decay, gradient_clipping
"""

        return f"""You are a Critic Agent analyzing the target model's weaknesses.

## Your Task
Read the target model ({self.model_name}) summary and identify concrete weaknesses and limitations.
{f"Also analyze the config file for potential hyperparameter issues." if config_path else ""}
{self.iteration_context_section}

## Domain Context
- Model: {self.model_name}
- Domain: {self.domain}

## Input
- Target Model Summary: {target_summary_path}
{config_section}

## Output
Write a markdown file to: {output_path}

## Instructions
1. Read the target model summary using read_file tool
{f"2. Read the config file using read_file tool" if config_path else ""}
2. Identify 3-5 concrete, actionable weaknesses (architecture + config)
3. For each weakness:
   - Specify the affected component ({self.components_display}) OR config parameter
   - Explain the limitation clearly
   - Cite evidence from the summary or config
   - State what improvement is needed

## Output Format (Markdown)
```markdown
# Weakness of Target Model: {self.model_name}

## Architecture Weaknesses

### Weakness 1: [Brief Title]
- **Component**: [{self.components_display}]
- **Limitation**: [Detailed explanation of the weakness]
- **Evidence**: [Quote or reference from the summary]
- **Improvement Needed**: [What capability should be added]

### Weakness 2: [Brief Title]
...

{f'''## Config Weaknesses (ONLY "# Can touch" sections)

### Config Issue 1: [Brief Title]
- **Parameter**: [e.g., encoder.architecture.hidden_dim]
- **Current Value**: [Current value]
- **Problem**: [Why this might be suboptimal]
- **Suggested Change**: [What value might be better and why]

### Config Issue 2: [Brief Title]
...

NOTE: The following sections are marked "# Never touch" and MUST NOT be modified:
- data (base_path, datatype, preprocessing, spatial_network)
- clustering (method, n_clusters)
- evaluation (metrics)
- output (result_path, checkpoint_dir, embedding_key, save_reconstruction)
- logging (level, log_file)
''' if config_path else ""}

## Summary of Critical Weaknesses
1. [First critical weakness]
2. [Second critical weakness]
3. [Third critical weakness]
```

IMPORTANT:
- Use write_file tool to save the output
- Be specific and actionable
- Focus on architectural and methodological limitations related to {self.domain}
{f"- For config: ONLY suggest changes to '# Can touch' sections" if config_path else ""}
- Content must be in English
"""

    # =========================================================================
    # ARTICLE RESEARCHER PROMPT
    # =========================================================================

    def get_article_researcher_prompt(
        self,
        weakness_path: str,
        output_path: str,
        papers_dir: str,
    ) -> str:
        """Generate prompt for searching and downloading papers.

        NOTE: Keywords are now generated by LLM based on summary.md, weakness.md, and code analysis.
        """
        return f"""You are an Article Researcher Agent.

## Goal
Find the top 2 relevant papers (2022+)
that can improve {self.model_name}'s {self.domain} architecture.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}

## Inputs
- Weakness Analysis: {weakness_path}

## Tool: search_filter_rank_download

This single tool does EVERYTHING automatically:
1. Searches papers from high-impact journals (2022+)
2. Filters by keywords in title/abstract
3. Checks each paper for GitHub repository links
4. Scores and ranks papers by relevance
5. Downloads top 2 papers to the output directory
6. Returns selected_papers with pdf_path included

### Usage
```
search_filter_rank_download(
    output_dir="{papers_dir}",
    start_year=2022,
    num_papers=2
)
```

### Returns
{{
  "success": true,
  "total_searched": 2000,
  "after_keyword_filter": 100,
  "with_github": 20,
  "num_downloaded": 2,
  "selected_papers": [
    {{
      "title": "...",
      "venue": "...",
      "year": 2024,
      "keywords_found": ["neural network", "..."],
      "relevance_score": 8,
      "pdf_url": "...",
      "pdf_path": "/path/to/downloaded.pdf",
      "github_urls": ["https://github.com/..."]
    }}
  ]
}}

## Process
1. Call search_filter_rank_download with output_dir="{papers_dir}"
2. Take the selected_papers from the result
3. Use write_file to save to: {output_path}

## Output JSON Format
{{
  "selected_papers": [<copy selected_papers from tool result>]
}}

IMPORTANT: Just call search_filter_rank_download ONCE - it handles search, filter, rank, AND download!
"""

    # =========================================================================
    # PROPOSAL GENERATION PROMPT
    # =========================================================================

    def get_proposal_generation_prompt(
        self,
        paper_name: str,
        weakness_path: str,
        paper_summary_path: str,
        target_summary_path: str,
        output_path: str,
    ) -> str:
        """Generate prompt for creating improvement proposal."""
        return f"""You are an Article Researcher Agent generating an improvement proposal.

## Your Task
Read the target model's weaknesses and your reference paper summary, then propose concrete improvements.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Components: {self.components_choice}

## Input Files
- Weakness Analysis: {weakness_path}
- Reference Paper Summary: {paper_summary_path}
- Target Model Summary: {target_summary_path}

## Output
Write a markdown file to: {output_path}

## Instructions
1. Read all input files using read_file tool
2. Identify which weaknesses your reference paper can address
3. Propose specific improvements based on the reference paper's methods
4. Be concrete about:
   - Which component to modify ({self.components_choice})
   - What architecture/method from the reference paper to apply
   - How it addresses the identified weaknesses
   - Implementation approach

## Output Format (Markdown)
```markdown
# Proposal Based on Reference Paper: {paper_name}

## 1. Weaknesses Addressed
List the specific weaknesses from the weakness analysis that this proposal addresses:
- Weakness [X]: [Brief description]
- Weakness [Y]: [Brief description]

## 2. Proposed Solution

### Target Component
[{self.components_choice}] - Choose ONE

### Key Method from Reference Paper
[Name and brief description of the method/architecture]

### How It Addresses Weaknesses
For each weakness addressed:
- **Weakness [X]**: [How the proposed method solves this]
- **Weakness [Y]**: [How the proposed method solves this]

## 3. Technical Details

### Architecture Overview
[High-level description of the proposed architecture]

### Key Components
- Component 1: [Description]
- Component 2: [Description]
- ...

### Input/Output Specifications
- Input: [Format and dimensions]
- Output: [Format and dimensions]

## 4. Implementation Approach

### Required Changes
1. [First change needed]
2. [Second change needed]
...

### Integration with Existing {self.model_name}
[How this integrates with other {self.model_name} components]

### Hyperparameters
- Parameter 1: [Suggested value and rationale]
- Parameter 2: [Suggested value and rationale]

## 5. Expected Benefits
1. [First expected improvement]
2. [Second expected improvement]
...

## 6. Potential Challenges
1. [First potential challenge and mitigation]
2. [Second potential challenge and mitigation]
```

IMPORTANT:
- Use write_file tool to save the proposal
- Be specific and implementable
- Focus on ONE clear proposal that best addresses {self.model_name}'s weaknesses
- Provide concrete technical details
- Content must be in English
"""

    # =========================================================================
    # OTHER PAPER READER PROMPT
    # =========================================================================

    def get_other_paper_reader_prompt(
        self,
        pdf_path: str,
        output_path: str,
        repos_dir: str,
    ) -> str:
        """Generate prompt for reading reference papers."""
        return f"""You are a Reference Paper Expert Agent.

## Your Task
Read the paper PDF and create a comprehensive summary focusing on the METHOD section,
specifically looking for architectures that could improve {self.model_name}'s {self.domain}.

## PDF Location
{pdf_path}

## Instructions
1. Use the read_pdf tool to read the paper
2. Use extract_github_urls to find any GitHub repository links in the paper
3. If GitHub URL found:
   - Use clone_github_repo(github_url, target_dir="{repos_dir}") to clone the repository
   - Use list_repo_structure to understand the codebase
   - IMPORTANT: Find and read encoder/model files (look for: model.py, encoder.py, layers.py, modules.py, net.py)
   - Use read_github_file to read the FULL implementation code
   - Copy the COMPLETE encoder class code into the summary (not snippets, FULL code)
4. Focus on extracting:
   - Novel neural network architectures
   - Encoding/embedding mechanisms
   - Attention mechanisms (if any)
   - How the model handles input features
   - Representation learning strategies
   - Any components applicable to {self.domain}

5. Write a structured markdown summary with ACTUAL CODE from GitHub

## Output Format
Write a markdown file with the following structure:

```markdown
# Reference Paper Summary

## Overview
[Brief description of the paper's main contribution]

## Core Architecture

### Model Layers
- Layer type: [describe the architecture]
- Information flow mechanism: [how information is processed]
- Aggregation function: [if applicable]
- Key operations: [main computational steps]

### Key Innovations
[What makes this architecture different/better]

## Transferability to {self.model_name}

### For {self.components[0].replace('_', ' ').title()}
- Applicability: [how this could be used]
- Required modifications: [what changes would be needed]
- Expected benefits: [why this would improve {self.model_name}]

### For Other Components
[If applicable to other components: {self.components_choice}]

## Implementation Details
- Input requirements: [input format and features]
- Output dimensions: [how output dim is determined]
- Key hyperparameters: [important params to tune]

## CRITICAL: Detailed Implementation Guide for Coding
This section is the MOST IMPORTANT. A developer will read ONLY this section to implement the model.
You MUST extract enough detail from the paper so someone can code it WITHOUT reading the paper.

### 1. Core Principle (Plain English)
Explain in 2-3 sentences:
- What is the key innovation?
- How does it differ from standard approaches?
- What problem does it solve?

### 2. Mathematical Formulation
Extract the EXACT equations from the paper:
- Forward pass equations
- Loss functions (if relevant)
- Key transformations

### 3. GitHub Code Paths (CRITICAL)
If you cloned the GitHub repository, list ALL file paths for encoder/decoder related code.
DO NOT copy code here - just list the paths so Code Expert can read them later.

#### Repository Info
- Repository URL: [url]
- Clone Path: [local path where repo was cloned]

#### Encoder/Model Files
List all files related to the encoder/model with their imports:
```
File: [full path to file]
  - Classes: [class names with line numbers]
  - Imports from: [other files this imports from]
```

### 4. Recommended Target Component
Based on this paper, which {self.model_name} component should be modified?
{chr(10).join([f"- [ ] {comp.replace('_', ' ').title()}" for comp in self.components])}

Reason: [why this component]

## Potential Benefits for {self.model_name}
[Concrete ways this could address {self.model_name}'s limitations in {self.domain}]
```

IMPORTANT: List file PATHS only, not code. The Code Expert will read the files directly.

Save the summary to: {output_path}
"""

    # =========================================================================
    # RANKING PROMPTS
    # =========================================================================

    def get_ranking_prompt(
        self,
        critique_path: str,
        proposal1_path: str,
        proposal2_path: str,
        target_summary_path: str,
        output_path: str,
    ) -> str:
        """Generate prompt for ranking proposals."""
        return f"""You are a Model Researcher Agent performing proposal ranking.

## Your Task
Read the critic's analysis and both proposals, then rank them based on feasibility, impact,
and alignment with {self.model_name}'s weaknesses.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Components: {self.components_choice}

## Input Files
- Critique of Proposals: {critique_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}
- Target Model Summary: {target_summary_path}

## Output
Write a markdown file to: {output_path}

## Instructions
1. Read all input files using read_file tool
2. Compare the two proposals based on:
   - **Feasibility**: How implementable is it? (code availability, integration complexity)
   - **Impact**: How well does it address the identified weaknesses?
   - **Technical Soundness**: Is the proposed architecture well-justified?
   - **Risk**: What are the implementation risks?
3. Consider the critic's feedback seriously
4. Rank the proposals as Rank #1 and Rank #2

## Output Format (Markdown)
```markdown
# Ranked Proposals - Round 1

## Evaluation Summary

### Proposal 1: [Title from proposal_other1.md]
**Feasibility Score**: [1-10]/10
- [Brief assessment]

**Impact Score**: [1-10]/10
- [Brief assessment]

**Technical Soundness Score**: [1-10]/10
- [Brief assessment]

**Risk Assessment**: [Low/Medium/High]
- [Key risks identified]

**Total Score**: [Sum]/30

---

### Proposal 2: [Title from proposal_other2.md]
**Feasibility Score**: [1-10]/10
- [Brief assessment]

**Impact Score**: [1-10]/10
- [Brief assessment]

**Technical Soundness Score**: [1-10]/10
- [Brief assessment]

**Risk Assessment**: [Low/Medium/High]
- [Key risks identified]

**Total Score**: [Sum]/30

---

## Final Ranking

### Rank #1: [Proposal X]
**Justification**:
1. [Primary reason for ranking first]
2. [Secondary reason]
3. [Tertiary reason]

**Key Strengths**:
- [Strength 1]
- [Strength 2]

**Concerns to Address**:
- [Concern 1 from critic's feedback]
- [Concern 2]

---

### Rank #2: [Proposal Y]
**Why Not First**:
1. [Primary reason for ranking second]
2. [Secondary reason]

**Potential for Improvement**:
- [What could make this proposal better]
```

IMPORTANT:
- Use write_file tool to save the ranking
- Be objective and evidence-based
- Consider both critic feedback and your own analysis
- Content must be in English
"""

    def get_final_ranking_prompt(
        self,
        critique_ranked_path: str,
        ranking_r1_path: str,
        proposal1_path: str,
        proposal2_path: str,
        output_path: str,
    ) -> str:
        """Generate prompt for final proposal selection."""
        return f"""You are a Model Researcher Agent making the final proposal selection.

## Your Task
Read the critic's feedback on your initial ranking and make the final decision on which proposal to implement.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}

## Input Files
- Critique of Ranked Proposals: {critique_ranked_path}
- Initial Ranking: {ranking_r1_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}

## Output
Write a markdown file to: {output_path}

## Instructions
1. Read all input files using read_file tool
2. Review the critic's concerns about your initial ranking
3. Decide whether to:
   - **Maintain** your initial Rank #1 choice (with updated justification)
   - **Change** to the other proposal (if critic's arguments are compelling)
4. Provide a clear, final decision with comprehensive justification

## Output Format (Markdown)
```markdown
# Final Rank #1 Proposal

## Decision
**Selected Proposal**: [Proposal X - Full Title]

**Decision Outcome**: [MAINTAINED initial ranking / CHANGED from initial ranking]

---

## Final Justification

### Primary Reasons for Selection
1. [Most important reason]
2. [Second most important reason]
3. [Third most important reason]

### Response to Critic's Concerns
The critic raised the following concerns about this proposal:
- **Concern 1**: [Critic's concern]
  - **Response**: [How this will be addressed or why it's acceptable]

- **Concern 2**: [Critic's concern]
  - **Response**: [How this will be addressed or why it's acceptable]

### Comparison with Alternative
Why this proposal is preferred over [Other Proposal]:
1. [Comparative advantage 1]
2. [Comparative advantage 2]
3. [Comparative advantage 3]

---

## Implementation Priority

### Must-Have Features
1. [Critical feature 1 from the proposal]
2. [Critical feature 2 from the proposal]

### Risk Mitigation Plan
- **Risk 1**: [Identified risk]
  - **Mitigation**: [How to handle it]

- **Risk 2**: [Identified risk]
  - **Mitigation**: [How to handle it]

### Success Criteria
1. [Criterion 1 - how to measure if implementation is successful]
2. [Criterion 2]
3. [Criterion 3]

---

## Next Steps for Implementation
1. [First implementation step]
2. [Second implementation step]
3. [Third implementation step]
```

IMPORTANT:
- Use write_file tool to save the final decision
- Be decisive - clearly state which proposal is Rank #1
- Address all of the critic's concerns
- Provide actionable next steps
- Content must be in English
"""

    # =========================================================================
    # CRITIQUE PROMPTS
    # =========================================================================

    def get_critique_proposals_prompt(
        self,
        weakness_path: str,
        proposal_paths: List[str],
    ) -> str:
        """Generate prompt for critiquing proposals."""
        proposals_list = "\n".join([f"- Proposal {i+1}: {path}" for i, path in enumerate(proposal_paths)])

        return f"""You are a Critic Agent providing critical analysis of improvement proposals.

## Your Task
Review all improvement proposals and provide thorough critical analysis of each.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Components: {self.components_choice}

## Input Files
- Weakness Analysis: {weakness_path}
{proposals_list}

## Instructions
1. Read the weakness analysis and all proposals using read_file tool
2. For EACH proposal, provide critique regardless of ranking
3. Identify:
   - Strengths of the proposal
   - Weaknesses and potential issues
   - Missing details or unclear points
   - Feasibility concerns

## Output Format
### Critique of Proposal 1: [Proposal Name]
**Strengths:**
- [Strength 1]
- [Strength 2]

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]

**Concerns:**
- [Concern 1]
- [Concern 2]

**Questions:**
- [Question 1]
- [Question 2]

---

### Critique of Proposal 2: [Proposal Name]
...

IMPORTANT: Be thorough and critical. Your goal is to find potential issues before implementation.
Focus on {self.domain}-specific considerations.
"""

    def get_critique_ranked_prompt(
        self,
        ranking_r1_path: str,
        proposal1_path: str,
        proposal2_path: str,
        weakness_path: str,
    ) -> str:
        """Generate prompt for critiquing ranked proposals."""
        return f"""You are a Critic Agent reviewing the Model Researcher's ranking decision.

## Your Task
Review the Model Researcher's ranking and challenge it if needed.

## Domain Context
- Target Model: {self.model_name}
- Domain: {self.domain}

## Input Files
- Initial Ranking: {ranking_r1_path}
- Proposal 1: {proposal1_path}
- Proposal 2: {proposal2_path}
- Weakness Analysis: {weakness_path}

## Instructions
1. Read all input files using read_file tool
2. Evaluate whether the ranking is justified:
   - Are the evaluation scores accurate?
   - Did the Model Researcher overlook any critical factors?
   - Is the Rank #1 choice truly the best option for {self.domain}?
3. Be critical but constructive
4. If you disagree with the ranking, explain why
5. Highlight any concerns that need to be addressed before implementation

## Output Format
### Ranking Review

**Overall Assessment**: [AGREE with ranking / PARTIALLY AGREE / DISAGREE]

**Evaluation of Scoring**:
- Feasibility scores: [Accurate / Overestimated / Underestimated]
- Impact scores: [Accurate / Overestimated / Underestimated]
- Technical soundness scores: [Accurate / Overestimated / Underestimated]

---

### Critique of Rank #1 Choice: [Proposal X]

**Do you agree this should be Rank #1?**: [YES / NO / WITH RESERVATIONS]

**Strengths of this decision**:
1. [Valid point in the ranking justification]
2. [Another valid point]

**Concerns about this decision**:
1. [Potential issue overlooked]
2. [Another concern]

**Critical questions before implementation**:
1. [Question 1 that must be answered]
2. [Question 2 that must be answered]

---

### Critique of Rank #2 Choice: [Proposal Y]

**Should this have been Rank #1 instead?**: [YES / NO / POSSIBLY]

**Undervalued strengths**:
1. [Strength that may have been underestimated]
2. [Another undervalued aspect]

---

### Additional Considerations

**Missing from the ranking analysis**:
1. [Factor 1 that should have been considered for {self.domain}]
2. [Factor 2 that should have been considered]

**Deal-breaker concerns** (if any):
- [Critical issue that could invalidate the chosen proposal]

IMPORTANT:
- Be critical but fair
- Focus on {self.domain}-specific technical concerns
- Help the Model Researcher make the best final decision
"""

    # =========================================================================
    # DEBATE ROUND PROMPTS (Dynamic Multi-Round Debate)
    # =========================================================================

    def get_debate_propose_prompt(
        self,
        paper_name: str,
        paper_summary_path: str,
        weakness_path: str,
        target_summary_path: str,
        current_round: int,
        config_path: str = None,
    ) -> str:
        """Generate prompt for proposing in debate round."""
        config_section = ""
        config_output_section = ""
        if config_path:
            config_section = f"""- Config File: {config_path}

IMPORTANT: The config file has `# Can touch` and `# Never touch` sections.
You may ONLY propose changes to `# Can touch` sections:
- encoder: hidden_dim, heads, dropout
- decoder: hidden_dim, heads, dropout
- training: epochs, learning_rate, weight_decay, gradient_clipping"""
            config_output_section = """
### Config Changes (ONLY "# Can touch" sections)
If your proposal requires config changes:
- **Parameter**: [e.g., encoder.architecture.hidden_dim]
- **Current → Proposed**: [10 → 512]
- **Reason**: [Why this change improves performance]

NOTE: Do NOT propose changes to "# Never touch" sections (data, clustering, evaluation, output, logging)"""

        return f"""You are the {paper_name} Expert participating in a debate to improve {self.model_name}.

## Your Task
Propose an improvement for {self.model_name} based on your reference paper's methodology.
Your proposal should include BOTH architecture changes AND config changes if needed.
{self.iteration_context_section}

## Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Your Reference Paper: {paper_name}
- Current Round: {current_round}

## Input Files
- Target Model Summary: {target_summary_path}
- Weakness Analysis: {weakness_path}
- Your Paper Summary: {paper_summary_path}
{config_section}

## Instructions
1. Read all input files{f" including the config file" if config_path else ""}
2. Identify which weaknesses your paper's method can address
3. Propose a clear, concrete improvement (architecture + config if needed)
4. Focus on {self.components_choice}

## Output Format
```markdown
## [{paper_name} Expert] Proposal - Round {current_round}

### Targeted Weakness
[Which weakness from the analysis you're addressing - include both architecture and config weaknesses]

### Proposed Solution
**Target Component**: [{self.components_choice}]

**Method**: [Name of method from your paper]

**How It Works**:
1. [Step 1]
2. [Step 2]
3. [Step 3]
{config_output_section}

### Expected Benefits
- [Benefit 1]
- [Benefit 2]

### Implementation Approach
[Brief description of how to implement - include both code changes and config changes]
```

Be confident but open to feedback. Your proposal will be critiqued and you may need to revise it.

IMPORTANT: DO NOT use write_file tool. Output the markdown content DIRECTLY in your response.
"""

    def get_debate_respond_and_revise_prompt(
        self,
        paper_name: str,
        other_paper_name: str,
        own_proposal: str,
        other_proposal: str,
        critic_feedback: str,
        current_round: int,
        config_path: str = None,
    ) -> str:
        """Generate prompt for responding to other proposal and revising.

        This prompt enables FULL ADOPTION - the expert can abandon their own
        proposal and support the other expert's proposal if convinced.
        """
        config_note = ""
        config_output_section = ""
        if config_path:
            config_note = f"""
NOTE: When revising proposals, you may also revise config changes.
Config file: {config_path}
Only modify "# Can touch" sections (encoder, decoder, training).
Do NOT modify "# Never touch" sections (data, clustering, evaluation, output, logging)."""
            config_output_section = """
#### Config Changes (if revised)
- **Parameter**: [e.g., encoder.architecture.hidden_dim]
- **Current → Proposed**: [value → new_value]
- **Reason**: [Why this change]"""

        return f"""You are the {paper_name} Expert participating in a debate to improve {self.model_name}.

## Your Task
Respond to the other expert's proposal and the Critic's feedback.
You may FULLY ADOPT the other proposal if you believe it's better.
{config_note}
{self.iteration_context_section}

## Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Your Paper: {paper_name}
- Other Paper: {other_paper_name}
- Current Round: {current_round}

## Your Proposal
{own_proposal}

## Other Expert's Proposal
{other_proposal}

## Critic's Feedback
{critic_feedback}

## Instructions
1. Consider the Critic's points carefully
2. Evaluate the other expert's proposal OBJECTIVELY
3. Choose ONE of the following options:

### Option A: MAINTAIN (with revisions)
- Explain why your approach is still better
- Address the Critic's concerns
- Revise your proposal to improve it (including config changes if needed)

### Option B: PARTIAL ADOPTION
- Acknowledge strengths in the other proposal
- Propose a HYBRID approach combining both
- Explain the merged benefits (including combined config changes)

### Option C: FULL ADOPTION (완전 수용)
- Acknowledge that the other proposal is SUPERIOR
- Explain WHY it better addresses {self.model_name}'s weaknesses
- Withdraw your proposal

## Output Format
```markdown
## [{paper_name} Expert] Response - Round {current_round}

### Decision: [MAINTAIN / PARTIAL_ADOPTION / FULL_ADOPTION]

### Response to Critic
[Address each of the Critic's concerns]

### Response to Other Expert
[Acknowledge valid points from {other_paper_name}'s proposal]

---

### If MAINTAIN:
#### Revised Proposal
[Updated proposal addressing concerns]
{config_output_section}

#### Why This Is Still Better
[Justification]

---

### If PARTIAL_ADOPTION:
#### Hybrid Approach
[Combined proposal integrating both methods]
{config_output_section}

#### Benefits of Integration
[Why combining is better than either alone]

---

### If FULL_ADOPTION:
#### Acknowledgment
[Why {other_paper_name}'s proposal is superior]

#### I WITHDRAW my proposal in favor of {other_paper_name}.

---
```

IMPORTANT:
- Be scientifically honest. If the other approach truly addresses the weaknesses better, adopt it.
- DO NOT use write_file tool. Output the markdown content DIRECTLY in your response.
- Your response will be saved automatically by the system.
"""

    def get_debate_ranking_prompt(
        self,
        all_proposals: str,
        critic_feedback: str,
        current_round: int,
    ) -> str:
        """Generate prompt for ranking proposals in debate."""
        return f"""You are the Model Researcher ranking proposals in the debate.

## Your Task
Rank all active proposals based on feasibility, impact, and technical soundness.

## Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Current Round: {current_round}

## Active Proposals
{all_proposals}

## Critic's Feedback
{critic_feedback}

## Instructions
1. Evaluate each proposal on:
   - **Feasibility** (1-10): How implementable is it?
   - **Impact** (1-10): How well does it address weaknesses?
   - **Technical Soundness** (1-10): Is the approach well-justified?
2. Consider the Critic's concerns
3. Rank proposals from best to worst

## Output Format
```markdown
## [Model Researcher] Ranking - Round {current_round}

### Proposal Scores

#### [Proposal 1 Name]
- Feasibility: X/10
- Impact: X/10
- Technical Soundness: X/10
- **Total: X/30**

#### [Proposal 2 Name]
- Feasibility: X/10
- Impact: X/10
- Technical Soundness: X/10
- **Total: X/30**

---

### Final Ranking
1. **Rank #1**: [Proposal Name] - [Total Score]
   - Key Strengths: [Brief]
   - Concerns: [Brief]

2. **Rank #2**: [Proposal Name] - [Total Score]
   - Key Strengths: [Brief]
   - Why Not #1: [Brief]
```

IMPORTANT: DO NOT use write_file tool. Output the markdown content DIRECTLY in your response.
"""

    def get_debate_final_selection_prompt(
        self,
        all_rounds_summary: str,
        final_proposals: str,
        final_critique: str,
        current_round: int,
        article1_status: str,
        article2_status: str,
    ) -> str:
        """Generate prompt for final selection with consensus detection."""
        return f"""You are the Model Researcher making the FINAL selection.

## Your Task
Based on ALL rounds of debate, select the FINAL implementation proposal.
Also determine if CONSENSUS has been reached.

## Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Total Rounds: {current_round}
- Article 1 Status: {article1_status}
- Article 2 Status: {article2_status}

## Debate Summary
{all_rounds_summary}

## Final Proposals
{final_proposals}

## Final Critique
{final_critique}

## Instructions
1. Review the entire debate history
2. Consider how proposals evolved across rounds
3. Note any FULL_ADOPTION declarations
4. Make a definitive selection
5. Assess consensus level

## Output Format
```markdown
## [Model Researcher] Final Selection - Round {current_round}

### Selected Proposal
**[Proposal Name]**

### Confidence Level
[0-100]% - [HIGH_CONFIDENCE / MEDIUM_CONFIDENCE / LOW_CONFIDENCE]

### Justification
1. [Primary reason]
2. [Secondary reason]
3. [Tertiary reason]

---

### Consensus Assessment

**Consensus Reached**: [YES / NO]

**Consensus Type**: [Choose one]
- STRONG_CONSENSUS: Both experts converged on similar solution
- FULL_ADOPTION: One expert fully adopted the other's proposal
- CONVERGENCE: Proposals merged into hybrid approach
- DOMINANT_WINNER: One proposal clearly superior
- NO_CONSENSUS: Still significant disagreement

**Consensus Reason**: [Explanation]

---

### If Consensus NOT Reached
#### Outstanding Issues
1. [Issue still debated]
2. [Another issue]

#### Recommendation
[Should we continue debate or proceed with current best?]

---

### Implementation Readiness
**Ready for Implementation**: [YES / NO]

**Blocker (if NO)**: [What needs to be resolved]

---

### FINAL_DECISION

I select **[Proposal Name]** as the implementation target.

[If high confidence, include: STRONG_CONSENSUS]
```

IMPORTANT:
- If consensus is clear, include "STRONG_CONSENSUS" or "FINAL_DECISION" in your response
- These keywords trigger automatic debate termination
- Be decisive but evidence-based
- DO NOT use write_file tool. Output the markdown content DIRECTLY in your response.
- Your response will be saved automatically by the system.
"""

    def get_debate_critique_round_prompt(
        self,
        proposals: str,
        current_round: int,
    ) -> str:
        """Generate prompt for critiquing proposals in a debate round."""
        return f"""You are the Critic providing feedback in the debate.

## Your Task
Critique ALL active proposals fairly and constructively.

## Context
- Target Model: {self.model_name}
- Domain: {self.domain}
- Current Round: {current_round}

## Proposals to Critique
{proposals}

## Instructions
1. Critique EACH proposal regardless of apparent ranking
2. Be thorough and constructive
3. Identify:
   - Strengths worth preserving
   - Weaknesses that need addressing
   - Missing details
   - Feasibility concerns for {self.domain}
4. Suggest specific improvements

## Output Format
```markdown
## [Critic] Feedback - Round {current_round}

### Critique of [Proposal 1 Name]

**Strengths:**
- [Strength 1]
- [Strength 2]

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]

**Missing Details:**
- [What needs clarification]

**Suggestions:**
- [Specific improvement 1]
- [Specific improvement 2]

---

### Critique of [Proposal 2 Name]

**Strengths:**
- [Strength 1]
- [Strength 2]

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]

**Missing Details:**
- [What needs clarification]

**Suggestions:**
- [Specific improvement 1]
- [Specific improvement 2]

---

### Overall Assessment
[Which proposal currently seems stronger and why]
[What would make the weaker proposal competitive]
```

Be critical but FAIR. Your goal is to help improve proposals, not to dismiss them.
Focus on {self.domain}-specific technical concerns.

IMPORTANT: DO NOT use write_file tool. Output the markdown content DIRECTLY in your response.
"""
