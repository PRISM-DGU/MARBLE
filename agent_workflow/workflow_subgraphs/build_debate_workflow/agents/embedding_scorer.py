"""Embedding-based Paper Scoring Module.

Implements the scoring formula:
S_total(t) = w_d × S_domain(t) + w_a × [β_t × S_arch(t) + (1 - β_t) × (1 - S_arch(t))]

Key features:
- Uses BAAI/bge-m3 for embeddings
- Momentum update for S_arch using previous iteration's selected paper score
- Dynamic w_d/w_a weights by iteration
- S_domain weight scheduling by iteration
- Agent-controlled beta (novelty coefficient)
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent_workflow.logger import logger
from agent_workflow.utils import get_project_root, get_free_gpu_device
from agent_workflow.evolving_memory import EvolvingMemory


# =============================================================================
# Configuration
# =============================================================================

# Default values (can be overridden via constructor)
DEFAULT_REWARD_BLOCK_SIZE = 10
DEFAULT_REWARD_WEIGHT = 0.1

# =============================================================================
# Global Singleton for Embedding Model (GPU 메모리 중복 로드 방지)
# =============================================================================

_GLOBAL_EMBEDDER = None
_GLOBAL_EMBEDDER_DEVICE = None


def _get_global_embedder(device: str):
    """모듈 레벨 싱글톤으로 BGEM3FlagModel 관리.

    iteration이 바뀌어도 같은 device면 모델 재사용.
    device가 바뀌면 기존 모델 해제 후 새로 로드.
    """
    global _GLOBAL_EMBEDDER, _GLOBAL_EMBEDDER_DEVICE

    if _GLOBAL_EMBEDDER is not None and _GLOBAL_EMBEDDER_DEVICE == device:
        logger.debug(f"[EmbeddingScorer] Reusing existing BGEM3FlagModel on {device}")
        return _GLOBAL_EMBEDDER

    # device가 바뀌었거나 처음 로드하는 경우
    if _GLOBAL_EMBEDDER is not None:
        logger.info(f"[EmbeddingScorer] Releasing old model from {_GLOBAL_EMBEDDER_DEVICE}")
        del _GLOBAL_EMBEDDER
        _GLOBAL_EMBEDDER = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        from FlagEmbedding import BGEM3FlagModel

        logger.info(f"[EmbeddingScorer] Loading BAAI/bge-m3 (FlagEmbedding) on {device}...")
        _GLOBAL_EMBEDDER = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            devices=[device] if device.startswith("cuda") else None
        )
        _GLOBAL_EMBEDDER_DEVICE = device
        logger.info(f"[EmbeddingScorer] BGEM3FlagModel loaded successfully on {device}")
        return _GLOBAL_EMBEDDER
    except ImportError:
        logger.error("[EmbeddingScorer] FlagEmbedding not installed. Run: pip install FlagEmbedding")
        raise


def cleanup_embedder() -> None:
    """embedding scoring 완료 후 GPU 메모리 해제.

    FlagEmbedding 모델을 CPU로 이동 후 삭제하고 CUDA 캐시를 정리합니다.
    paper_aggregator_node에서 scoring 완료 후 호출해야 합니다.
    """
    global _GLOBAL_EMBEDDER, _GLOBAL_EMBEDDER_DEVICE
    import gc

    if _GLOBAL_EMBEDDER is None:
        logger.debug("[EmbeddingScorer] No embedder to cleanup")
        return

    logger.info(f"[EmbeddingScorer] Cleaning up embedder from {_GLOBAL_EMBEDDER_DEVICE}...")

    try:
        import torch

        # 모델을 CPU로 이동 후 삭제
        if hasattr(_GLOBAL_EMBEDDER, 'model') and hasattr(_GLOBAL_EMBEDDER.model, 'to'):
            _GLOBAL_EMBEDDER.model.to('cpu')

        del _GLOBAL_EMBEDDER
        _GLOBAL_EMBEDDER = None
        _GLOBAL_EMBEDDER_DEVICE = None

        # 가비지 컬렉션
        gc.collect()

        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("[EmbeddingScorer] GPU memory released successfully")
    except Exception as e:
        logger.warning(f"[EmbeddingScorer] Error during cleanup: {e}")
        # 에러가 발생해도 글로벌 변수는 정리
        _GLOBAL_EMBEDDER = None
        _GLOBAL_EMBEDDER_DEVICE = None

MODEL_TO_DOMAIN_FOLDER = {
    "stagate": "spatial_transcriptomics",
    "deepst": "spatial_transcriptomics",
    "deeptta": "drug_response_prediction",
    "deepdr": "drug_response_prediction",
    "dlm-dti": "drug_target_interaction",
    "hyperattentiondti": "drug_target_interaction",
}

DOMAIN_DESCRIPTIONS = {
    "spatial_transcriptomics": (
        "Spatial transcriptomics deep learning for analyzing spatial gene expression patterns, "
        "tissue domain identification, single cell spatial analysis. "
        "Graph attention networks, variational autoencoders, deep embedded clustering for spatial data."
    ),
    "drug_response_prediction": (
        "Drug response prediction and drug sensitivity prediction using deep learning. "
        "IC50 prediction, cancer drug response modeling. "
        "Transformer molecular representation, graph isomorphism networks, SMILES encoding, gene expression encoders."
    ),
    "drug_target_interaction": (
        "Drug-target interaction prediction and drug discovery using deep learning. "
        "DTI prediction, compound-protein interaction modeling. "
        "Heterogeneous network embedding, knowledge graphs, pathway neural networks, random walk embeddings."
    ),
}


# =============================================================================
# Weight Schedules
# =============================================================================

def get_w_d(current_iteration: int, total_iterations: int) -> float:
    """Get domain weight w_d for given iteration.

    Dynamic schedule: w_d starts at 0.9 (iter 1) and decreases to 0.5 (final iter)
    Linear interpolation: w_d = 0.9 - 0.4 * progress
    """
    if total_iterations <= 1:
        return 0.9
    progress = (current_iteration - 1) / (total_iterations - 1)
    w_d = 0.9 - 0.4 * progress
    return round(max(0.5, min(0.9, w_d)), 2)


def get_w_a(current_iteration: int, total_iterations: int) -> float:
    """Get architecture weight w_a.

    Dynamic schedule: w_a starts at 0.1 (iter 1) and increases to 0.5 (final iter)
    w_a = 1 - w_d
    """
    return round(1.0 - get_w_d(current_iteration, total_iterations), 2)


# =============================================================================
# Main Scorer Class
# =============================================================================

class EmbeddingScorer:
    """Paper scorer using bge-m3 embeddings with momentum update.

    Momentum update for S_arch (변경됨):
        S_arch = 0.9 × Sim(E_best, E_method) + 0.1 × S_arch_2nd(paper_id)

    Where:
        - E_best: build_t 폴더의 코드 임베딩 (best iteration의 변경사항이 반영된 상태)
        - S_arch_2nd(paper_id): 2nd best iteration의 s_arch_scores.json에서 조회
    """

    def __init__(
        self,
        target_model: str,
        current_iteration: int,
        total_iterations: int,
        build_dir: Path,
        beta: float = 1.0,  # Ignored, always 1.0
        used_papers: Optional[List[str]] = None,
        reward_patience: int = DEFAULT_REWARD_BLOCK_SIZE,
        reward_weight: float = DEFAULT_REWARD_WEIGHT,
    ):
        """Initialize the embedding scorer.

        Args:
            target_model: Model name (e.g., 'stagate', 'deeptta')
            current_iteration: Current iteration number (1-based)
            total_iterations: Total number of iterations (for w_d/w_a scheduling)
            build_dir: Path to current build_N directory
            beta: DEPRECATED - Always fixed at 1.0 (parameter ignored)
            used_papers: List of previously used paper titles to exclude
            reward_patience: Reward block size (--patience flag, default 10)
            reward_weight: Reward weight (--weight flag, default 0.1)
        """
        self.target_model = target_model.lower()
        self.current_iteration = current_iteration
        self.total_iterations = max(1, total_iterations)
        self.build_dir = Path(build_dir)
        self.beta = 1.0  # Fixed at 1.0, parameter ignored
        self.used_papers = set(used_papers) if used_papers else set()

        # Reward settings (from CLI flags)
        self.reward_patience = reward_patience
        self.reward_weight = reward_weight

        # Domain configuration
        self.domain_folder = MODEL_TO_DOMAIN_FOLDER.get(
            self.target_model, "drug_response_prediction"
        )
        self.domain_description = DOMAIN_DESCRIPTIONS.get(self.domain_folder, "")

        # PDF directory
        self.pdf_dir = Path(get_project_root()) / "experiments" / "pdf" / self.domain_folder

        # Dynamic weights (w_d: 0.9→0.1, w_a: 0.1→0.9)
        self.w_d = get_w_d(current_iteration, total_iterations)
        self.w_a = get_w_a(current_iteration, total_iterations)

        # Lazy-loaded components
        self._embedder = None
        self._domain_embedding: Optional[np.ndarray] = None
        self._current_code_embedding: Optional[np.ndarray] = None  # E_best (현재 build_t)

        # 2nd best iteration의 S_arch scores (momentum용)
        self._2nd_s_arch_scores: Dict[str, float] = {}
        self._2nd_best_iteration: Optional[int] = None
        self._load_2nd_s_arch_scores()

        # Paper reward scores (cross-iteration learning)
        # Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
        self._paper_rewards: Dict[str, float] = {}
        self._reward_snapshot_iteration: Optional[int] = None
        self._load_paper_rewards()

        logger.info(f"[EmbeddingScorer] Initialized for {target_model} iter {current_iteration}/{total_iterations}")
        logger.info(f"[EmbeddingScorer] Domain: {self.domain_folder}")
        logger.info(f"[EmbeddingScorer] PDF dir: {self.pdf_dir}")
        logger.info(f"[EmbeddingScorer] Beta={self.beta}, 2nd_best_iter={self._2nd_best_iteration}, 2nd_s_arch_loaded={len(self._2nd_s_arch_scores)}")
        logger.info(f"[EmbeddingScorer] Reward settings: patience={self.reward_patience}, weight={self.reward_weight}")
        logger.info(f"[EmbeddingScorer] Paper rewards loaded: {len(self._paper_rewards)}")

    # =========================================================================
    # Embedding Model
    # =========================================================================

    def _get_embedder(self):
        """Lazy initialization of BGEM3FlagModel.

        Uses module-level singleton to prevent GPU memory accumulation
        across iterations. Model is loaded once and reused.

        See: https://huggingface.co/BAAI/bge-m3
        """
        if self._embedder is None:
            device = get_free_gpu_device()
            self._embedder = _get_global_embedder(device)
        return self._embedder

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to dense vector using BGEM3FlagModel.

        Args:
            text: Text to encode (will be truncated if too long)

        Returns:
            Dense embedding vector as numpy array
        """
        embedder = self._get_embedder()
        # BGEM3FlagModel returns dict with 'dense_vecs' for dense embeddings
        output = embedder.encode(
            [text],
            batch_size=1,
            max_length=8192,  # bge-m3 supports up to 8192 tokens
        )
        return output["dense_vecs"][0]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # =========================================================================
    # S_arch Score Persistence (for Momentum)
    # =========================================================================

    def _get_s_arch_scores_path(self, iteration: int) -> Path:
        """Get path for S_arch scores JSON file."""
        return self.build_dir.parent / f"build_{iteration}" / "s_arch_scores.json"

    def _load_2nd_s_arch_scores(self) -> None:
        """2nd best iteration의 S_arch scores 로드 (momentum용).

        memory.json에서 2nd best iteration 조회 후 해당 s_arch_scores.json 로드.
        iter 1에서는 momentum 없음.
        baseline(iter 0)에는 s_arch_scores.json이 없으므로 momentum 미적용.
        """
        if self.current_iteration <= 1:
            self._2nd_s_arch_scores = {}
            self._2nd_best_iteration = None
            return

        # EvolvingMemory에서 2nd best 조회
        workspace = Path(get_project_root()) / "experiments"
        mb = EvolvingMemory(workspace_path=str(workspace))
        second_best_iter = mb.get_2nd_best_iteration()

        if second_best_iter is None:
            logger.info(f"[EmbeddingScorer] No 2nd best iteration (momentum not applied)")
            self._2nd_s_arch_scores = {}
            self._2nd_best_iteration = None
            return

        # baseline(iter 0)에는 s_arch_scores.json이 없음
        if second_best_iter == 0:
            logger.info(f"[EmbeddingScorer] 2nd best = baseline (iter 0), no s_arch_scores (momentum not applied)")
            self._2nd_s_arch_scores = {}
            self._2nd_best_iteration = 0
            return

        self._2nd_best_iteration = second_best_iter

        # 2nd best의 s_arch_scores.json 로드
        scores_path = self._get_s_arch_scores_path(second_best_iter)
        if scores_path.exists():
            try:
                with open(scores_path, "r", encoding="utf-8") as f:
                    self._2nd_s_arch_scores = json.load(f)
                logger.info(f"[EmbeddingScorer] Loaded {len(self._2nd_s_arch_scores)} 2nd best S_arch scores from iter {second_best_iter}")
            except Exception as e:
                logger.warning(f"[EmbeddingScorer] Failed to load 2nd best S_arch scores: {e}")
                self._2nd_s_arch_scores = {}
        else:
            logger.info(f"[EmbeddingScorer] No S_arch scores file found at {scores_path}")
            self._2nd_s_arch_scores = {}

    def _load_paper_rewards(self) -> None:
        """EvolvingMemory에서 paper rewards 로드.

        reward_patience block 단위 reward 적용:
        - iter 1~patience: reward 미적용
        - iter patience+1 ~ 2*patience: iter patience 스냅샷 적용
        - ...
        예: patience=5 → iter 6-10에서 iter 5 스냅샷 적용
        """
        prev_block_end = ((self.current_iteration - 1) // self.reward_patience) * self.reward_patience

        # 첫 블록은 reward 미적용
        if prev_block_end == 0:
            self._paper_rewards = {}
            self._reward_snapshot_iteration = None
            logger.info(
                f"[EmbeddingScorer] iter {self.current_iteration}: no reward applied (1~{self.reward_patience})"
            )
            return

        try:
            workspace = Path(get_project_root()) / "experiments"
            mb = EvolvingMemory(workspace_path=str(workspace))
            snapshot = mb.get_paper_reward_snapshot(prev_block_end)

            # 새 블록 시작 여부 확인
            is_new_block_start = (self.current_iteration == prev_block_end + 1)

            if snapshot is None:
                if is_new_block_start:
                    snapshot = mb.save_paper_reward_snapshot(prev_block_end)
                    self._reward_snapshot_iteration = prev_block_end
                else:
                    snapshot = mb.get_all_paper_rewards()
                    self._reward_snapshot_iteration = None
                    logger.warning(
                        f"[EmbeddingScorer] iter {self.current_iteration}: no snapshot (iter {prev_block_end}), using accumulated rewards"
                    )
            else:
                self._reward_snapshot_iteration = prev_block_end

            # 새 블록 시작 시 눈에 띄는 로그
            if is_new_block_start:
                block_end = prev_block_end + self.reward_patience
                logger.info("=" * 60)
                logger.info(f"[REWARD APPLIED] New block started! iter {self.current_iteration}")
                logger.info(f"  - Applying iter {prev_block_end} snapshot")
                logger.info(f"  - Block range: iter {self.current_iteration}~{block_end}")
                logger.info(f"  - Papers: {len(snapshot)}")
                logger.info(f"  - patience={self.reward_patience}, weight={self.reward_weight}")
                logger.info("=" * 60)
            else:
                logger.info(
                    f"[EmbeddingScorer] iter {self.current_iteration}: reward snapshot loaded (iter {prev_block_end}, {len(snapshot)} papers)"
                )

            self._paper_rewards = snapshot
        except Exception as e:
            logger.warning(f"[EmbeddingScorer] Failed to load paper rewards: {e}")
            self._paper_rewards = {}
            self._reward_snapshot_iteration = None

    def save_s_arch_scores(self, scores: Dict[str, float]) -> None:
        """Save current iteration's S_arch scores for future momentum use.

        Args:
            scores: Dict mapping paper_id (title digest) to S_arch score
        """
        save_path = self._get_s_arch_scores_path(self.current_iteration)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)
            logger.info(f"[EmbeddingScorer] Saved {len(scores)} S_arch scores to {save_path}")
        except Exception as e:
            logger.error(f"[EmbeddingScorer] Failed to save S_arch scores: {e}")

    def _get_paper_id(self, paper: Dict[str, Any]) -> str:
        """Generate unique ID for paper (based on title digest)."""
        title = paper.get("title", "")
        normalized = title.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    # =========================================================================
    # Domain Embedding
    # =========================================================================

    def _get_domain_embedding(self) -> np.ndarray:
        """Get embedding for domain description."""
        if self._domain_embedding is None:
            domain_text = f"""
            Model: {self.target_model}
            Domain: {self.domain_folder.replace('_', ' ')}
            Description: {self.domain_description}
            """
            self._domain_embedding = self._encode(domain_text.strip())
            logger.debug(f"[EmbeddingScorer] Computed domain embedding")
        return self._domain_embedding

    # =========================================================================
    # Code Embedding
    # =========================================================================

    def _get_code_embedding(self) -> np.ndarray:
        """현재 build_t 폴더의 코드 임베딩 (E_best).

        build_t에는 setup_build_workspace_node에서 best iteration의 변경사항이
        이미 반영되어 있으므로, 단순히 현재 build_dir 전체를 읽으면 됨.

        For iter 1: build_1 (baseline 템플릿 기반)
        For iter 2+: build_t (build_0 + best의 변경사항이 적용된 상태)
        """
        if self._current_code_embedding is None:
            code_text = self._load_code_text_for_current_iteration()
            self._current_code_embedding = self._encode(code_text[:8000])
            logger.info(f"[EmbeddingScorer] Computed code embedding ({len(code_text)} chars)")
        return self._current_code_embedding

    def _load_code_text_for_current_iteration(self) -> str:
        """현재 build_t 폴더에서 코드 텍스트 로드.

        iter 2+에서는 best iteration의 변경사항이 이미 build_t에 적용되어 있으므로
        단순히 현재 build_dir를 읽으면 best의 코드를 임베딩할 수 있음.
        """
        code_parts = []

        # 항상 현재 build_dir에서 로드 (best 변경사항이 이미 반영됨)
        code_parts.extend(self._load_all_code_from_build(self.build_dir))

        # Fallback
        if not code_parts:
            code_parts.append(f"""
            Model: {self.target_model}
            Architecture: graph neural network, encoder, decoder,
            molecular representation, gene expression, attention mechanism
            """)
            logger.warning(f"[EmbeddingScorer] Using fallback code description")

        return "\n\n".join(code_parts)

    def _load_all_code_from_build(self, build_dir: Path) -> List[str]:
        """Load all code from build directory (src + components + config)."""
        code_parts = []

        # 1. Load src/*.py
        src_dir = build_dir / "src"
        if src_dir.exists():
            for py_file in list(src_dir.glob("*.py"))[:10]:
                try:
                    content = py_file.read_text(encoding="utf-8")[:3000]
                    code_parts.append(f"# src/{py_file.name}\n{content}")
                except Exception:
                    continue

        # 2. Load components/*.py
        components_dir = build_dir / "components"
        if components_dir.exists():
            for py_file in list(components_dir.glob("*.py")):
                try:
                    content = py_file.read_text(encoding="utf-8")[:3000]
                    code_parts.append(f"# components/{py_file.name}\n{content}")
                except Exception:
                    continue

        # 3. Load config.yaml
        config_path = build_dir / "config.yaml"
        if config_path.exists():
            try:
                content = config_path.read_text(encoding="utf-8")[:2000]
                code_parts.append(f"# config.yaml\n{content}")
            except Exception:
                pass

        logger.debug(f"[EmbeddingScorer] Loaded {len(code_parts)} files from {build_dir}")
        return code_parts

    # =========================================================================
    # PDF Processing
    # =========================================================================

    def _extract_pdf_text(self, pdf_path: str, max_chars: int = 15000) -> str:
        """Extract full text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            full_text = " ".join(text_parts)
            full_text = re.sub(r"\s+", " ", full_text).strip()
            return full_text[:max_chars]
        except Exception as e:
            logger.warning(f"[EmbeddingScorer] PDF extract failed for {pdf_path}: {e}")
            return ""

    def _extract_method_section(self, pdf_text: str, max_chars: int = 5000) -> str:
        """Extract Method/Methodology section from PDF text (keyword-based)."""
        if not pdf_text:
            return ""

        pdf_lower = pdf_text.lower()

        # Method section start patterns
        method_patterns = [
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(method(?:ology)?|approach|proposed\s+method|our\s+method|model(?:\s+architecture)?|architecture|technical\s+approach)\s*(?:\n|\.)",
        ]

        # Next section patterns
        next_section_patterns = [
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(experiment|result|evaluation|implementation|dataset|conclusion|discussion|related\s+work)\s*(?:\n|\.)",
        ]

        # Find method section start
        method_start = -1
        for pattern in method_patterns:
            match = re.search(pattern, pdf_lower, re.IGNORECASE)
            if match:
                method_start = match.start()
                break

        if method_start == -1:
            return ""

        # Find next section (end of method)
        method_end = len(pdf_text)
        remaining_text = pdf_lower[method_start + 50:]

        for pattern in next_section_patterns:
            match = re.search(pattern, remaining_text, re.IGNORECASE)
            if match:
                method_end = method_start + 50 + match.start()
                break

        method_text = pdf_text[method_start:method_end]
        method_text = re.sub(r"\s+", " ", method_text).strip()

        return method_text[:max_chars]

    # =========================================================================
    # Paper Loading
    # =========================================================================

    def load_papers(self) -> List[Dict[str, Any]]:
        """Load papers from local PDF directory's all_candidates.json."""
        if not self.pdf_dir.exists():
            logger.error(f"[EmbeddingScorer] PDF directory not found: {self.pdf_dir}")
            return []

        # Load from all_candidates.json
        candidates_path = self.pdf_dir / "all_candidates.json"
        if candidates_path.exists():
            try:
                with open(candidates_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                papers = data.get("papers", [])
                logger.info(f"[EmbeddingScorer] Loaded {len(papers)} papers from {candidates_path}")
                return papers
            except Exception as e:
                logger.error(f"[EmbeddingScorer] Failed to load candidates: {e}")
                return []

        # Fallback: try downloaded_metadata.json
        metadata_path = self.pdf_dir / "downloaded_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                papers = data.get("downloaded_papers", [])
                logger.info(f"[EmbeddingScorer] Loaded {len(papers)} papers from metadata")
                return papers
            except Exception as e:
                logger.error(f"[EmbeddingScorer] Failed to load metadata: {e}")

        return []

    def _get_pdf_path(self, paper: Dict[str, Any], index: int) -> Optional[str]:
        """Get PDF file path for a paper."""
        # Check if pdf_path already exists
        if paper.get("pdf_path") and os.path.exists(paper["pdf_path"]):
            return paper["pdf_path"]

        # Try to construct path from naming convention: {source}_{index:04d}_{safe_title}.pdf
        source = paper.get("source", "pmc")
        title = paper.get("title", f"paper_{index}")
        safe_title = re.sub(r"[^\w\s\-]", "", title)
        safe_title = re.sub(r"\s+", "_", safe_title.strip())[:60]

        # Try exact match
        expected_name = f"{source}_{index:04d}_{safe_title}.pdf"
        expected_path = self.pdf_dir / expected_name
        if expected_path.exists():
            return str(expected_path)

        # Try glob match
        pattern = f"{source}_{index:04d}_*.pdf"
        matches = list(self.pdf_dir.glob(pattern))
        if matches:
            return str(matches[0])

        return None

    # =========================================================================
    # Scoring
    # =========================================================================

    def score_paper(self, paper: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Score a single paper using the formula.

        S_total = w_d × S_domain + w_a × [β × S_arch + (1 - β) × novelty]

        Where S_arch uses momentum (2nd best iteration 기준):
        S_arch = 0.9 × Sim(E_best, E_method) + 0.1 × S_arch_2nd(paper_id)
        """
        # Get embeddings
        domain_emb = self._get_domain_embedding()
        code_emb = self._get_code_embedding()  # E_best (best 변경사항이 반영된 build_t)

        # Get paper ID for momentum lookup
        paper_id = self._get_paper_id(paper)

        # Get abstract (for domain scoring)
        abstract = paper.get("abstract", "")
        if not abstract:
            abstract = paper.get("title", "")

        # Get method section (for architecture scoring)
        pdf_path = self._get_pdf_path(paper, index)
        method_text = ""
        pdf_read_success = False

        if pdf_path:
            pdf_text = self._extract_pdf_text(pdf_path)
            if pdf_text:
                pdf_read_success = True
                method_text = self._extract_method_section(pdf_text)

        # Fallback to abstract if no method section
        if not method_text or len(method_text) < 100:
            method_text = abstract

        # Compute paper embeddings using _encode (BGEM3FlagModel)
        abstract_emb = self._encode(abstract[:2000])
        method_emb = self._encode(method_text[:4000])

        # 1. Domain similarity: Sim(E_abstract, E_domain)
        S_domain = self._cosine_similarity(abstract_emb, domain_emb)

        # 2. Architecture similarity with MOMENTUM (2nd best 기준)
        # S_arch = 0.9 × Sim(E_best, E_method) + 0.1 × S_arch_2nd
        current_arch_sim = self._cosine_similarity(method_emb, code_emb)

        # 2nd best의 저장된 S_arch 점수 조회
        s_arch_2nd = self._2nd_s_arch_scores.get(paper_id)

        if s_arch_2nd is not None and self.current_iteration > 1:
            # Apply momentum: 0.9 * current + 0.1 * 2nd_best
            S_arch = 0.9 * current_arch_sim + 0.1 * s_arch_2nd
            momentum_applied = True
        else:
            # iter 1 or no 2nd best score: no momentum
            S_arch = current_arch_sim
            momentum_applied = False

        # 3. Novelty = 1 - S_arch
        novelty = 1.0 - S_arch

        # 4. Architecture component with beta
        # arch_component = β × S_arch + (1 - β) × novelty
        arch_component = self.beta * S_arch + (1 - self.beta) * novelty

        # 5. Final score
        # S_total = w_d × S_domain + w_a × arch_component
        S_total = self.w_d * S_domain + self.w_a * arch_component

        # 6. Apply paper reward (after first block, using snapshot)
        # V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
        reward = self._paper_rewards.get(paper_id, 0.0)
        S_total = S_total + reward

        return {
            **paper,
            "paper_id": paper_id,
            "scores": {
                "domain": round(S_domain, 4),
                "architecture_current": round(current_arch_sim, 4),
                "architecture_2nd": round(s_arch_2nd, 4) if s_arch_2nd is not None else None,
                "architecture": round(S_arch, 4),  # After momentum
                "novelty": round(novelty, 4),
                "arch_component": round(arch_component, 4),
                "reward": round(reward, 4),  # Paper reward (cross-iteration learning)
                "final": round(S_total, 4),
            },
            "scoring_params": {
                "w_d": self.w_d,
                "w_a": self.w_a,
                "beta": self.beta,
                "momentum_applied": momentum_applied,
                "momentum_source": f"2nd_best_iter_{self._2nd_best_iteration}" if momentum_applied else None,
                "reward_applied": self.current_iteration > self.reward_patience,
                "reward_snapshot_iteration": self._reward_snapshot_iteration,
                "iteration": self.current_iteration,
                "total_iterations": self.total_iterations,
            },
            "pdf_path": pdf_path,
            "pdf_read_success": pdf_read_success,
            "abstract_length": len(abstract),
            "method_length": len(method_text),
        }

    def score_all_papers(self) -> Tuple[List[Dict], Dict]:
        """Score all papers and return sorted results.

        Also saves S_arch scores for momentum in next iteration.

        Returns:
            Tuple of (scored_papers sorted by final score, metadata)
        """
        papers = self.load_papers()
        if not papers:
            logger.error("[EmbeddingScorer] No papers to score")
            return [], {}

        # NOTE: Duplicate filtering DISABLED - same papers can be selected across iterations
        # This allows the system to revisit promising papers with different beta/weights

        logger.info(f"[EmbeddingScorer] Scoring {len(papers)} papers...")

        scored_papers = []
        s_arch_scores = {}  # For saving momentum data

        for i, paper in enumerate(papers):
            try:
                scored = self.score_paper(paper, i)
                scored_papers.append(scored)

                # Collect S_arch for saving (momentum for next iter)
                paper_id = scored.get("paper_id")
                if paper_id:
                    s_arch_scores[paper_id] = scored["scores"]["architecture"]

            except Exception as e:
                logger.warning(f"[EmbeddingScorer] Failed to score paper {i}: {e}")
                continue

            if (i + 1) % 20 == 0:
                logger.info(f"[EmbeddingScorer] Progress: {i+1}/{len(papers)}")

        # Sort by final score (descending)
        scored_papers.sort(key=lambda x: x["scores"]["final"], reverse=True)

        # Save S_arch scores for momentum in next iteration
        self.save_s_arch_scores(s_arch_scores)

        metadata = {
            "target_model": self.target_model,
            "domain": self.domain_folder,
            "iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "total_papers": len(papers),
            "scored_papers": len(scored_papers),
            "weights": {
                "w_d": self.w_d,
                "w_a": self.w_a,
                "beta": self.beta,
            },
            "momentum": {
                "enabled": self.current_iteration > 1,
                "2nd_best_iteration": self._2nd_best_iteration,
                "2nd_scores_loaded": len(self._2nd_s_arch_scores),
                "formula": "S_arch = 0.9 * Sim(E_best, method) + 0.1 * S_arch_2nd",
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Log top 5
        logger.info(f"[EmbeddingScorer] ========== Top 5 Papers ==========")
        for i, p in enumerate(scored_papers[:5], 1):
            scores = p["scores"]
            momentum_str = f", 2nd={scores['architecture_2nd']:.3f}" if scores.get('architecture_2nd') else ""
            logger.info(
                f"#{i} (final={scores['final']:.3f}, domain={scores['domain']:.3f}, "
                f"arch={scores['architecture']:.3f}{momentum_str}): {p.get('title', 'N/A')[:50]}..."
            )

        return scored_papers, metadata

    def score_paper_with_weights(
        self, paper: Dict[str, Any], index: int, w_d: float, w_a: float
    ) -> Dict[str, Any]:
        """Score a single paper with custom weights.

        Uses the same formula but with provided w_d and w_a instead of instance values.
        S_total = w_d × S_domain + w_a × [β × S_arch + (1 - β) × novelty]
        """
        # Get embeddings (cached)
        domain_emb = self._get_domain_embedding()
        code_emb = self._get_code_embedding()

        paper_id = self._get_paper_id(paper)

        # Get abstract
        abstract = paper.get("abstract", "")
        if not abstract:
            abstract = paper.get("title", "")

        # Get method section
        pdf_path = self._get_pdf_path(paper, index)
        method_text = ""
        pdf_read_success = False

        if pdf_path:
            pdf_text = self._extract_pdf_text(pdf_path)
            if pdf_text:
                pdf_read_success = True
                method_text = self._extract_method_section(pdf_text)

        if not method_text or len(method_text) < 100:
            method_text = abstract

        # Compute embeddings
        abstract_emb = self._encode(abstract[:2000])
        method_emb = self._encode(method_text[:4000])

        # Domain similarity
        S_domain = self._cosine_similarity(abstract_emb, domain_emb)

        # Architecture similarity with momentum
        current_arch_sim = self._cosine_similarity(method_emb, code_emb)
        s_arch_2nd = self._2nd_s_arch_scores.get(paper_id)

        if s_arch_2nd is not None and self.current_iteration > 1:
            S_arch = 0.9 * current_arch_sim + 0.1 * s_arch_2nd
            momentum_applied = True
        else:
            S_arch = current_arch_sim
            momentum_applied = False

        # Novelty and arch component (beta is fixed at 1.0)
        novelty = 1.0 - S_arch
        arch_component = self.beta * S_arch + (1 - self.beta) * novelty

        # Final score with custom weights
        S_total = w_d * S_domain + w_a * arch_component

        # Apply paper reward (after first block, using snapshot)
        reward = self._paper_rewards.get(paper_id, 0.0)
        S_total = S_total + reward

        return {
            **paper,
            "paper_id": paper_id,
            "scores": {
                "domain": round(S_domain, 4),
                "architecture_current": round(current_arch_sim, 4),
                "architecture_2nd": round(s_arch_2nd, 4) if s_arch_2nd is not None else None,
                "architecture": round(S_arch, 4),
                "novelty": round(novelty, 4),
                "arch_component": round(arch_component, 4),
                "reward": round(reward, 4),
                "final": round(S_total, 4),
            },
            "scoring_params": {
                "w_d": w_d,
                "w_a": w_a,
                "beta": self.beta,
                "momentum_applied": momentum_applied,
                "momentum_source": f"2nd_best_iter_{self._2nd_best_iteration}" if momentum_applied else None,
                "reward_applied": self.current_iteration > self.reward_patience,
                "reward_snapshot_iteration": self._reward_snapshot_iteration,
                "iteration": self.current_iteration,
                "total_iterations": self.total_iterations,
            },
            "pdf_path": pdf_path,
            "pdf_read_success": pdf_read_success,
            "abstract_length": len(abstract),
            "method_length": len(method_text),
        }

    def score_all_papers_stratified(self) -> Tuple[Dict[str, Dict], Dict]:
        """Score all papers with 3 different weight configurations.

        OPTIMIZED: Embeddings are computed once per paper, then final scores
        are calculated for each weight configuration using cached S_domain and S_arch.

        Returns Top-20 for each configuration:
        - high_domain (w_d=0.9, w_a=0.1): Domain relevance focused
        - balanced (w_d=0.5, w_a=0.5): Equal weight
        - high_arch (w_d=0.1, w_a=0.9): Architecture similarity focused

        Returns:
            Tuple of (stratified_results dict, metadata dict)
        """
        papers = self.load_papers()
        if not papers:
            logger.error("[EmbeddingScorer] No papers to score")
            return {}, {}

        WEIGHT_CONFIGS = [
            {"w_d": 0.9, "w_a": 0.1, "label": "high_domain", "select_count": 2},
            {"w_d": 0.5, "w_a": 0.5, "label": "balanced", "select_count": 1},
            {"w_d": 0.1, "w_a": 0.9, "label": "high_arch", "select_count": 2},
        ]

        logger.info(f"[EmbeddingScorer] Stratified scoring {len(papers)} papers...")
        logger.info(f"[EmbeddingScorer] Phase 1: Computing embeddings (once per paper)...")

        # =================================================================
        # PHASE 1: Compute embeddings ONCE per paper
        # =================================================================
        base_scored_papers = []
        s_arch_scores = {}

        for i, paper in enumerate(papers):
            try:
                # Use existing score_paper() which computes embeddings
                scored = self.score_paper(paper, i)
                base_scored_papers.append(scored)

                # Collect S_arch for momentum
                paper_id = scored.get("paper_id")
                if paper_id:
                    s_arch_scores[paper_id] = scored["scores"]["architecture"]

            except Exception as e:
                logger.warning(f"[EmbeddingScorer] Failed to score paper {i}: {e}")
                continue

            if (i + 1) % 20 == 0:
                logger.info(f"[EmbeddingScorer] Progress: {i+1}/{len(papers)}")

        logger.info(f"[EmbeddingScorer] Phase 1 complete: {len(base_scored_papers)} papers scored")

        # Save S_arch scores for momentum in next iteration
        self.save_s_arch_scores(s_arch_scores)

        # =================================================================
        # PHASE 2: Calculate final scores for each weight config (no re-embedding)
        # =================================================================
        logger.info(f"[EmbeddingScorer] Phase 2: Calculating final scores for 3 weight configs...")

        stratified_results = {}

        for config in WEIGHT_CONFIGS:
            w_d = config["w_d"]
            w_a = config["w_a"]
            label = config["label"]

            # Recalculate final score with different weights (reuse S_domain, S_arch)
            scored_for_config = []
            for base_paper in base_scored_papers:
                paper_copy = base_paper.copy()
                scores = base_paper["scores"].copy()

                # Recalculate: S_total = w_d * S_domain + w_a * S_arch + reward (beta=1.0)
                reward = scores.get("reward", 0.0)
                S_total = w_d * scores["domain"] + w_a * scores["architecture"] + reward
                scores["final"] = round(S_total, 4)

                paper_copy["scores"] = scores
                paper_copy["scoring_params"] = {
                    **base_paper.get("scoring_params", {}),
                    "w_d": w_d,
                    "w_a": w_a,
                }
                paper_copy["weight_label"] = label
                scored_for_config.append(paper_copy)

            # Sort by final score
            scored_for_config.sort(key=lambda x: x["scores"]["final"], reverse=True)

            # Store Top-20
            stratified_results[label] = {
                "top_20": scored_for_config[:20],
                "select_count": config["select_count"],
                "weights": {"w_d": w_d, "w_a": w_a},
                "total_scored": len(scored_for_config),
            }

            # Log top 3 for this config
            logger.info(f"[EmbeddingScorer] === {label} Top 3 ===")
            for i, p in enumerate(scored_for_config[:3], 1):
                scores = p["scores"]
                logger.info(
                    f"  #{i} (final={scores['final']:.3f}, domain={scores['domain']:.3f}, "
                    f"arch={scores['architecture']:.3f}): {p.get('title', 'N/A')[:50]}..."
                )

        metadata = {
            "target_model": self.target_model,
            "domain": self.domain_folder,
            "iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "total_papers": len(papers),
            "scored_papers": len(base_scored_papers),
            "weight_configs": WEIGHT_CONFIGS,
            "beta": self.beta,
            "momentum": {
                "enabled": self.current_iteration > 1,
                "2nd_best_iteration": self._2nd_best_iteration,
                "2nd_scores_loaded": len(self._2nd_s_arch_scores),
            },
            "optimization": "embeddings computed once, final scores recalculated per weight config",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[EmbeddingScorer] Stratified scoring complete. 3 Top-20 lists generated.")
        return stratified_results, metadata

    def save_stratified_results(
        self,
        stratified_results: Dict[str, Any],
        metadata: Dict,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save stratified scoring results to JSON file."""
        if output_path is None:
            output_path = self.build_dir / "embeddings" / "stratified_scoring.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            "stratified_results": stratified_results,
            "metadata": metadata,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[EmbeddingScorer] Stratified results saved to {output_path}")
        return output_path

    # =========================================================================
    # Result Saving
    # =========================================================================

    def save_results(
        self,
        scored_papers: List[Dict],
        metadata: Dict,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save scoring results to JSON file."""
        if output_path is None:
            output_path = self.build_dir / "embeddings" / "scoring_result.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            "top_20": scored_papers[:20],
            "all_scored": scored_papers,
            "metadata": metadata,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[EmbeddingScorer] Results saved to {output_path}")
        return output_path

    def save_selected_paper(self, selected_paper: Dict, output_path: Optional[Path] = None) -> Path:
        """Save selected paper info for tracking used papers in future iterations."""
        if output_path is None:
            output_path = self.build_dir / "embeddings" / "selected_paper.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "iteration": self.current_iteration,
            "selected_paper_title": selected_paper.get("title", "Unknown"),
            "scores": selected_paper.get("scores", {}),
            "github_urls": selected_paper.get("github_urls", []),
            "source": selected_paper.get("source", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"[EmbeddingScorer] Selected paper saved: {data['selected_paper_title'][:50]}...")
        return output_path

# =============================================================================
# Utility Functions
# =============================================================================

def load_used_papers(build_dir: Path, current_iteration: int) -> List[str]:
    """Load list of previously used paper titles from all previous iterations.

    Args:
        build_dir: Current build directory (build_N)
        current_iteration: Current iteration number

    Returns:
        List of paper titles that were used in previous iterations
    """
    used_papers = []

    for prev_iter in range(1, current_iteration):
        prev_build_dir = build_dir.parent / f"build_{prev_iter}"
        selected_paper_path = prev_build_dir / "embeddings" / "selected_paper.json"

        if selected_paper_path.exists():
            try:
                with open(selected_paper_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("selected_paper_title")
                if title:
                    used_papers.append(title)
            except Exception:
                continue

    if used_papers:
        logger.info(f"[EmbeddingScorer] Found {len(used_papers)} previously used papers")

    return used_papers
