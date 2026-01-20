"""Build Debate Workflow Prompts

This module provides prompts for the build debate workflow:
- Static prompts: Legacy hardcoded prompts for backwards compatibility
- PromptFactory: Dynamic prompt generation based on model configuration
"""

# Legacy static prompts (for backwards compatibility)
from .build_debate_prompts import (
    PAPER_READER_DEEPTTA_PROMPT,
    PAPER_READER_VISION_PROMPT,
    PAPER_READER_OTHER_PROMPT,
    CRITIC_ROUND1_PROMPT,
    CRITIC_ROUND2_PROMPT,
    PAPER_RESPONSE_PROMPT,
    PROPOSAL_AGENT_PROMPT,
    MODEL_PROBLEM_PROMPT,
    ARTICLE_RESEARCHER_PROMPT,
    DEBATE_EXPERT_PROMPT,
    # New dynamic prompts
    CRITIC_WEAKNESS_PROMPT,
    ARTICLE_RESEARCHER_PROPOSAL_PROMPT,
    MODEL_RESEARCHER_RANKING_PROMPT,
    MODEL_RESEARCHER_FINAL_RANKING_PROMPT,
    CRITIC_RANKED_CRITIQUE_PROMPT,
    CRITIC_FINAL_PROMPT,
)

# Dynamic prompt factory for model-specific prompts
from .prompt_factory import PromptFactory

__all__ = [
    # Legacy static prompts
    "PAPER_READER_DEEPTTA_PROMPT",
    "PAPER_READER_VISION_PROMPT",
    "PAPER_READER_OTHER_PROMPT",
    "CRITIC_ROUND1_PROMPT",
    "CRITIC_ROUND2_PROMPT",
    "PAPER_RESPONSE_PROMPT",
    "PROPOSAL_AGENT_PROMPT",
    "MODEL_PROBLEM_PROMPT",
    "ARTICLE_RESEARCHER_PROMPT",
    "DEBATE_EXPERT_PROMPT",
    "CRITIC_WEAKNESS_PROMPT",
    "ARTICLE_RESEARCHER_PROPOSAL_PROMPT",
    "MODEL_RESEARCHER_RANKING_PROMPT",
    "MODEL_RESEARCHER_FINAL_RANKING_PROMPT",
    "CRITIC_RANKED_CRITIQUE_PROMPT",
    "CRITIC_FINAL_PROMPT",
    # Dynamic prompt factory
    "PromptFactory",
]
