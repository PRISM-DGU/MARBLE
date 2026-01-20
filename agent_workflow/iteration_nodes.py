"""Iteration 워크플로우를 위한 노드들.

노드 구성:
- init_iteration: iteration 초기화, EvolvingMemory 생성
- inject_memory_context: 이전 iteration 컨텍스트 로드
- save_to_memory: 현재 iteration 결과 저장
- check_continue: 다음 iteration 진행 여부 결정
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_workflow.logger import logger
from agent_workflow.state import MARBLEState
from agent_workflow.evolving_memory import EvolvingMemory, IterationAnalyzerAgent
from agent_workflow.utils import get_project_root


def get_build_path(iteration: int) -> Path:
    """iteration별 build 폴더 경로 반환.

    Args:
        iteration: iteration 번호 (1부터 시작)

    Returns:
        experiments/build_{iteration} 경로
    """
    return Path(get_project_root()) / "experiments" / f"build_{iteration}"


def _parse_implementation_proposal(proposal_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """implementation_proposal.md에서 변경 내용 파싱.

    Args:
        proposal_path: implementation_proposal.md 파일 경로

    Returns:
        (component, description) 튜플
        - component: 변경할 컴포넌트 (encoder, decoder 등)
        - description: 변경 내용 설명
    """
    if not proposal_path.exists():
        logger.warning(f"[ITERATION] Proposal 파일 없음: {proposal_path}")
        return None, None

    try:
        content = proposal_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[ITERATION] Proposal 파일 읽기 실패: {e}")
        return None, None

    component = None
    description = None

    # 1. Component 추출 (여러 패턴 시도)
    component_patterns = [
        r"Component to Modify[:\s]*\*?\*?([a-zA-Z_]+)",  # ## 2. Component to Modify: **encoder**
        r"Target Component[:\s]*\*?\*?([a-zA-Z_]+)",
        r"modify the[:\s]*\*?\*?([a-zA-Z_]+)",
        r"변경.*컴포넌트[:\s]*\*?\*?([a-zA-Z_]+)",
    ]
    for pattern in component_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            component = match.group(1).lower().strip("*").strip()
            break

    # 2. Description 추출 (Decision Summary 또는 Architecture 섹션에서)
    description_parts = []

    # Decision Summary에서 핵심 내용 추출
    decision_match = re.search(
        r"Decision Summary[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if decision_match:
        decision_text = decision_match.group(1).strip()
        # 첫 번째 의미 있는 문장 추출 (불릿 포인트 또는 일반 텍스트)
        lines = [l.strip().lstrip("-*").strip() for l in decision_text.split("\n") if l.strip()]
        if lines:
            description_parts.append(lines[0][:200])  # 최대 200자

    # Architecture 변경 내용 추출
    arch_match = re.search(
        r"(?:New Architecture|Architecture|Proposed Architecture)[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if arch_match:
        arch_text = arch_match.group(1).strip()
        lines = [l.strip().lstrip("-*").strip() for l in arch_text.split("\n") if l.strip()]
        if lines and lines[0] not in description_parts:
            description_parts.append(lines[0][:100])

    # Config Changes 요약 추가
    config_match = re.search(
        r"Config Changes[^\n]*\n+(.*?)(?=\n##|\n---|\Z)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if config_match:
        config_text = config_match.group(1).strip()
        # Parameter 변경 내용 추출
        param_matches = re.findall(r"\*?\*?Parameter\*?\*?[:\s]*([^\n]+)", config_text, re.IGNORECASE)
        if param_matches:
            description_parts.append(f"Config: {param_matches[0][:50]}")

    if description_parts:
        description = " | ".join(description_parts)
    else:
        # fallback: 파일의 처음 150자
        clean_content = re.sub(r"[#\*\-]+", "", content[:500]).strip()
        if clean_content:
            description = clean_content[:150] + "..."

    logger.info(f"[ITERATION] Proposal 파싱: component={component}, description={description[:50] if description else None}...")
    return component, description


def _extract_proposal_sections(proposal_path: Path) -> Optional[str]:
    """implementation_proposal.md에서 핵심 섹션 추출.

    추출 섹션:
    - ## 1. Decision Summary
    - ## 2. Architecture Overview
    - ## 6. Config Changes (IMPORTANT!)

    제외:
    - ## 3. Code to Implement (코드 블록)
    - ## 4, 5 등 기타 섹션

    Args:
        proposal_path: implementation_proposal.md 파일 경로

    Returns:
        추출된 섹션 문자열 (없으면 None)
    """
    if not proposal_path.exists():
        logger.warning(f"[ITERATION] Proposal 파일 없음: {proposal_path}")
        return None

    try:
        content = proposal_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[ITERATION] Proposal 파일 읽기 실패: {e}")
        return None

    sections = []

    # Decision Summary 추출
    decision_match = re.search(
        r"(##\s*1\.?\s*Decision Summary.*?)(?=##\s*2\.|##\s*3\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if decision_match:
        sections.append(decision_match.group(1).strip())

    # Architecture Overview 추출
    arch_match = re.search(
        r"(##\s*2\.?\s*Architecture Overview.*?)(?=##\s*3\.|##\s*4\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if arch_match:
        sections.append(arch_match.group(1).strip())

    # Config Changes 추출 (## 6. Config Changes)
    config_match = re.search(
        r"(##\s*6\.?\s*Config Changes.*?)(?=##\s*7\.|##\s*8\.|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    if config_match:
        sections.append(config_match.group(1).strip())

    if sections:
        result = "\n\n".join(sections)
        logger.info(f"[ITERATION] Proposal 섹션 추출: {len(result)} chars")
        return result

    return None


def _extract_weakness_summary(weakness_path: Path) -> Optional[str]:
    """weakness_of_target_model.md에서 핵심 내용 추출.

    전체 파일 내용을 반환 (일반적으로 크기가 작음).

    Args:
        weakness_path: weakness_of_target_model.md 파일 경로

    Returns:
        파일 전체 내용 (없으면 None)
    """
    if not weakness_path.exists():
        logger.warning(f"[ITERATION] Weakness 파일 없음: {weakness_path}")
        return None

    try:
        content = weakness_path.read_text(encoding="utf-8")
        logger.info(f"[ITERATION] Weakness 파일 추출: {len(content)} chars")
        return content.strip()
    except Exception as e:
        logger.warning(f"[ITERATION] Weakness 파일 읽기 실패: {e}")
        return None


def _extract_paper_titles(other_papers_path: Path) -> List[str]:
    """other_papers.json에서 논문 title 추출.

    Args:
        other_papers_path: other_papers.json 파일 경로

    Returns:
        논문 title 리스트
    """
    if not other_papers_path.exists():
        logger.warning(f"[ITERATION] other_papers.json 없음: {other_papers_path}")
        return []

    try:
        data = json.loads(other_papers_path.read_text(encoding="utf-8"))
        titles = [p["title"] for p in data.get("selected_papers", []) if p.get("title")]
        logger.info(f"[ITERATION] 논문 title 추출: {len(titles)}개")
        return titles
    except Exception as e:
        logger.warning(f"[ITERATION] other_papers.json 파싱 실패: {e}")
        return []


def init_iteration_node(state: MARBLEState) -> Dict[str, Any]:
    """Iteration 초기화 노드.

    일반 모드 (--task build):
    - EvolvingMemory 초기화 (기존 데이터 clear)
    - current_iteration = 1로 설정
    - total_iterations 확인
    - build_1 ~ build_N 폴더 미리 생성

    Continue 모드 (--task continue):
    - EvolvingMemory 유지 (clear 하지 않음)
    - current_iteration = 사용자 지정 값 (--iter N)
    - 기존 build_N 폴더 유지 (삭제하지 않음)
    - 지정된 iteration 폴더만 생성 (없으면)

    Returns:
        초기화된 iteration 관련 state 필드들
    """
    total_iterations = state.get("total_iterations", 1)
    target_model = state.get("target_model", "unknown")
    is_continue_mode = state.get("is_continue_mode", False)
    start_iteration = state.get("current_iteration", 1) if is_continue_mode else 1
    reward_patience = state.get("reward_patience", 10)
    reward_weight = state.get("reward_weight", 0.1)

    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    if is_continue_mode:
        # Continue 모드: 기존 데이터 유지, EvolvingMemory에서 세션 정보 읽기
        session_info = mb.get_session_info()
        planned = session_info.get("planned_iterations")
        saved_model = session_info.get("target_model")
        completed = session_info.get("completed_iterations", 0)

        if planned is None:
            # EvolvingMemory에 세션 정보가 없으면 경고 (이전 --task build 실행 기록 없음)
            logger.warning(f"[ITERATION] Continue 모드: EvolvingMemory에 세션 정보 없음. 단일 iteration으로 진행")
            planned = start_iteration  # 지정된 iter만 실행

        # total_iterations를 EvolvingMemory에서 읽은 값으로 설정
        actual_total = planned
        logger.info(f"[ITERATION] Continue 모드: iteration {start_iteration}부터 재개")
        logger.info(f"[ITERATION] 원래 계획: {planned}회, 완료: {completed}회, 모델: {saved_model or target_model}")
        logger.info(f"[ITERATION] EvolvingMemory 유지, 기존 폴더 보존")

        # Reward 설정: CLI에서 지정하면 업데이트, 아니면 기존 값 유지
        existing_reward = mb.get_reward_settings()
        if reward_patience != 10 or reward_weight != 0.1:
            # CLI에서 명시적으로 지정된 경우 업데이트
            mb.set_reward_settings(patience=reward_patience, weight=reward_weight)
            logger.info(f"[ITERATION] Continue 모드: Reward 설정 업데이트 (patience={reward_patience}, weight={reward_weight})")
        else:
            # 기존 설정 사용
            reward_patience = existing_reward.get("patience", 10)
            reward_weight = existing_reward.get("weight", 0.1)
            logger.info(f"[ITERATION] Continue 모드: 기존 Reward 설정 사용 (patience={reward_patience}, weight={reward_weight})")

        # 지정된 iteration 폴더가 없으면 생성
        build_path = get_build_path(start_iteration)
        build_path.mkdir(parents=True, exist_ok=True)
        (build_path / "build_debate_outputs").mkdir(exist_ok=True)
        (build_path / "src").mkdir(exist_ok=True)
        logger.info(f"[ITERATION] build_{start_iteration} 폴더 준비 완료")

        return {
            "current_iteration": start_iteration,
            "total_iterations": actual_total,  # EvolvingMemory에서 읽은 원래 계획 횟수
            "iteration_context": str(mb.memory_file) if start_iteration > 1 else "",
            "reward_patience": reward_patience,
            "reward_weight": reward_weight,
            "processing_logs": [f"[ITERATION] Continue 모드: iteration {start_iteration}/{actual_total}부터 재개"],
        }
    else:
        # 일반 모드: 처음부터 시작
        logger.info(f"[ITERATION] 초기화: {total_iterations}회 반복 예정 (모델: {target_model})")

        # EvolvingMemory 초기화 (새 세션 시작)
        mb.clear()  # 이전 데이터 초기화

        # 베이스라인 설정 (iter0)
        mb.init_baseline(target_model)

        # 세션 정보 저장 (continue 모드에서 사용)
        mb.set_session_info(planned_iterations=total_iterations, target_model=target_model)

        # Reward 설정 저장 (--patience, --weight 플래그)
        mb.set_reward_settings(patience=reward_patience, weight=reward_weight)

        # 기존 build_N 폴더들 삭제 (build_0 포함)
        for i in range(0, 100):  # build_0부터 충분히 큰 범위로 기존 폴더 정리
            old_build = get_build_path(i)
            if old_build.exists():
                shutil.rmtree(str(old_build))
                logger.info(f"[ITERATION] 기존 폴더 삭제: {old_build}")
            elif i > 0:
                break  # build_1 이후는 연속되어야 하므로, 없으면 중단

        # build_0 생성 (baseline 템플릿, 절대 수정하지 않음)
        build_0_path = get_build_path(0)
        source_path = Path(get_project_root()) / "docker_images" / target_model
        if source_path.exists():
            shutil.copytree(
                str(source_path),
                str(build_0_path),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
            )
            logger.info(f"[ITERATION] build_0 생성 완료 (baseline 템플릿): {source_path} → {build_0_path}")
        else:
            logger.warning(f"[ITERATION] docker_images/{target_model} 없음 - build_0 생성 스킵")

        # build_1 ~ build_N 폴더 생성 (각각 독립적인 workspace)
        for i in range(1, total_iterations + 1):
            build_path = get_build_path(i)
            build_path.mkdir(parents=True, exist_ok=True)
            (build_path / "build_debate_outputs").mkdir(exist_ok=True)
            (build_path / "src").mkdir(exist_ok=True)
            logger.info(f"[ITERATION] build_{i} 폴더 생성 완료")

        return {
            "current_iteration": 1,
            "total_iterations": total_iterations,
            "iteration_context": "",  # 첫 iteration은 컨텍스트 없음
            "reward_patience": reward_patience,
            "reward_weight": reward_weight,
            "processing_logs": [f"[ITERATION] 초기화 완료: {total_iterations}회 반복, build_1~{total_iterations} 생성"],
        }


def inject_memory_context_node(state: MARBLEState) -> Dict[str, Any]:
    """이전 iteration 컨텍스트 주입 노드.

    EvolvingMemory에서 이전 iteration 정보를 로드하여
    iteration_context에 저장 (프롬프트 주입용).

    iter 2+에서는 memory.json 요약을 로그로 출력:
    - 완료된 iteration 수
    - 최고 성능 iteration
    - 직전 iteration 결과
    - 누적 교훈

    Returns:
        iteration_context가 포함된 state 업데이트
    """
    current_iteration = state.get("current_iteration", 1)

    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    logs = [f"[ITERATION] Iteration {current_iteration} 시작"]

    # 첫 iteration이 아니면 메모리 요약 출력
    if current_iteration > 1:
        context = str(mb.memory_file)
        logger.info(f"[ITERATION] 메모리 파일 경로: {context}")

        # EvolvingMemory 요약 출력
        session_info = mb.get_session_info()
        completed = session_info.get("completed_iterations", 0)
        planned = session_info.get("planned_iterations", 0)

        logs.append(f"[MEMORY] 진행 상황: {completed}/{planned} iterations 완료")

        # 최고 성능 정보
        if mb.data.best_iteration is not None and mb.data.best_performance:
            best_perf = mb.data.best_performance
            perf_str = []
            if best_perf.rmse is not None:
                perf_str.append(f"RMSE={best_perf.rmse:.4f}")
            if best_perf.pearson is not None:
                perf_str.append(f"Pearson={best_perf.pearson:.4f}")
            # custom_metrics (spatial models)
            if best_perf.custom_metrics:
                for k, v in best_perf.custom_metrics.items():
                    if v is not None:
                        perf_str.append(f"{k}={v:.4f}")
            if perf_str:
                logs.append(f"[MEMORY] 최고 성능: iter {mb.data.best_iteration} ({', '.join(perf_str)})")
                logger.info(f"[ITERATION] Best: iter {mb.data.best_iteration} ({', '.join(perf_str)})")

        # 직전 iteration 결과
        last_record = mb.get_last_iteration()
        if last_record:
            if last_record.analysis.improved is True:
                result_str = "✅ 개선"
            elif last_record.analysis.improved is False:
                result_str = "❌ 하락"
            else:
                result_str = "➖ baseline"
            logs.append(f"[MEMORY] 직전 iter {last_record.iteration}: {result_str}")
            if last_record.analysis.reason:
                logs.append(f"[MEMORY] 이유: {last_record.analysis.reason[:100]}...")
            logger.info(f"[ITERATION] Last iter {last_record.iteration}: {result_str}")

        # 누적 교훈 (최근 3개)
        if mb.data.key_lessons:
            recent_lessons = mb.data.key_lessons[-3:]
            logs.append(f"[MEMORY] 핵심 교훈 ({len(mb.data.key_lessons)}개 중 최근 3개):")
            for i, lesson in enumerate(recent_lessons, 1):
                logs.append(f"  {i}. {lesson[:80]}...")
                logger.info(f"[ITERATION] Lesson {i}: {lesson[:50]}...")

        # 실패한 접근법 경고
        if mb.data.failed_approaches:
            logs.append(f"[MEMORY] ⚠️ 피해야 할 접근법: {len(mb.data.failed_approaches)}개")
            for approach in mb.data.failed_approaches[-2:]:
                logs.append(f"  - {approach[:60]}...")

        logger.info(f"[ITERATION] Memory context 로드 완료: {completed} iterations")
    else:
        context = ""
        logger.info("[ITERATION] 첫 iteration - 컨텍스트 없음")

    return {
        "iteration_context": context,
        "processing_logs": logs,
    }


async def save_to_memory_node(state: MARBLEState) -> Dict[str, Any]:
    """현재 iteration 결과를 EvolvingMemory에 저장하는 노드.

    1. 메트릭 수집 (iteration_metrics)
    2. 변경 사항 수집 (target_component, 변경 설명)
    3. IterationAnalyzerAgent로 분석 (reason, lessons 생성)
    4. EvolvingMemory에 저장

    Returns:
        저장 완료 상태
    """
    current_iteration = state.get("current_iteration", 1)
    target_model = state.get("target_model", "unknown")

    # 메트릭 수집
    current_metrics = state.get("iteration_metrics", {})
    docker_success = state.get("docker_test_success", False)
    docker_output = state.get("docker_test_output", "")

    # 변경 사항 수집: implementation_proposal.md에서 파싱
    target_component = state.get("target_component", "unknown")
    build_path = get_build_path(current_iteration)
    proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
    weakness_path = build_path / "build_debate_outputs" / "weakness_of_target_model.md"
    other_papers_path = build_path / "build_debate_outputs" / "other_papers.json"
    parsed_component, parsed_description = _parse_implementation_proposal(proposal_path)

    # 사용된 논문 title 추출
    papers_used = _extract_paper_titles(other_papers_path)

    # md 파일에서 상세 내용 추출
    implementation = _extract_proposal_sections(proposal_path)
    weakness = _extract_weakness_summary(weakness_path)

    # 파싱된 값 우선, 없으면 state에서 가져온 값 사용
    changes = {
        "component": parsed_component or target_component or "unknown",
        "description": parsed_description or f"Iteration {current_iteration} 변경",
        "implementation": implementation,
        "weakness": weakness,
    }

    logger.info(f"[ITERATION] 결과 저장 시작: iteration {current_iteration}")
    logger.info(f"[ITERATION] 메트릭: {current_metrics}")
    logger.info(f"[ITERATION] Docker 성공: {docker_success}")

    # EvolvingMemory 로드
    workspace = Path(get_project_root()) / "experiments"
    mb = EvolvingMemory(workspace_path=str(workspace))

    # best iteration 메트릭 가져오기 (delta 계산용)
    best_metrics = None
    if mb.data.best_performance:
        perf_dict = mb.data.best_performance.model_dump()
        best_metrics = {
            k: v for k, v in perf_dict.items()
            if v is not None and k != "custom_metrics"
        }
        # custom_metrics도 포함 (spatial models: ari, nmi, silhouette)
        custom = perf_dict.get("custom_metrics", {})
        if custom:
            for k, v in custom.items():
                if v is not None:
                    best_metrics[k] = v
        logger.info(f"[ITERATION] Delta 계산 기준: best iteration {mb.data.best_iteration}")

    # IterationAnalyzerAgent로 분석 (best 대비 delta 계산)
    try:
        analyzer = IterationAnalyzerAgent()
        primary_metrics = None
        if target_model:
            model_metric = EvolvingMemory.MODEL_PRIMARY_METRIC.get(target_model.lower())
            if model_metric:
                primary_metrics = [model_metric]
        analysis = analyzer.analyze(
            current_iteration=current_iteration,
            current_metrics=current_metrics,
            current_changes=changes,
            prev_metrics=best_metrics,  # best 기준으로 delta 계산
            docker_output=docker_output,
            primary_metrics=primary_metrics,
        )
        analysis_dict = analysis.model_dump()
    except Exception as e:
        logger.warning(f"[ITERATION] 분석 실패: {e}")
        analysis_dict = {
            "improved": None,
            "delta": None,
            "reason": f"분석 실패: {e}",
            "lessons": [],
        }

    # Docker 실패 시 improved = False로 강제 설정
    # (메트릭이 없어서 improved = None이 되는 경우 처리)
    if not docker_success and analysis_dict.get("improved") is None:
        analysis_dict["improved"] = False
        analysis_dict["reason"] = f"Docker test failed after max retries. {analysis_dict.get('reason', '')}"
        logger.warning(f"[ITERATION] Docker 실패 → improved = False로 설정")

    # EvolvingMemory에 저장 (iteration별 build 폴더 경로)
    artifacts = {
        "debate_outputs_path": f"experiments/build_{current_iteration}/build_debate_outputs",
        "src_path": f"experiments/build_{current_iteration}/src",
    }

    mb.add_iteration(
        iteration=current_iteration,
        performance=current_metrics,
        changes=changes,
        analysis=analysis_dict,
        artifacts=artifacts,
        weights=None,  # Stratified scoring uses fixed weights (0.9/0.1, 0.5/0.5, 0.1/0.9)
        papers_used=papers_used,
    )

    # Paper reward 업데이트: iteration 결과에 따라 논문별 reward 반영
    # Formula: V_i = Sim_i + (w × (N_success - N_failure) / (N_total + 1))
    mb.update_paper_rewards(
        papers_used=papers_used,
        iteration=current_iteration,
        improved=analysis_dict.get("improved")
    )

    improved_str = "✅ 개선" if analysis_dict.get("improved") else (
        "❌ 하락" if analysis_dict.get("improved") is False else "➖ baseline"
    )

    return {
        "processing_logs": [
            f"[ITERATION] Iteration {current_iteration} 저장 완료",
            f"[ITERATION] 결과: {improved_str}",
        ],
    }


def check_continue_node(state: MARBLEState) -> Dict[str, Any]:
    """다음 iteration 진행 여부 결정 노드.

    current_iteration < total_iterations 이면 다음 iteration으로,
    아니면 END.

    Returns:
        다음 iteration 준비 또는 종료 상태
    """
    current_iteration = state.get("current_iteration", 1)
    total_iterations = state.get("total_iterations", 1)

    if current_iteration < total_iterations:
        next_iter = current_iteration + 1
        logger.info(f"[ITERATION] 다음 iteration으로: {next_iter}/{total_iterations}")

        return {
            "current_iteration": next_iter,
            # Docker 테스트 관련 상태 리셋
            "docker_test_success": False,
            "docker_test_error": None,
            "docker_test_round": 0,
            "docker_test_output": None,
            "iteration_metrics": {},
            # Continue stage 리셋 (다음 iteration은 debate부터 시작)
            "continue_stage": None,
            "processing_logs": [f"[ITERATION] Iteration {next_iter}/{total_iterations} 준비"],
        }
    else:
        logger.info(f"[ITERATION] 모든 iteration 완료: {total_iterations}회")
        return {
            # route에서 end로 가도록 current_iteration을 total+1로 설정
            "current_iteration": total_iterations + 1,
            "processing_logs": [f"[ITERATION] 모든 {total_iterations}회 iteration 완료"],
        }


def route_after_save_to_memory(state: MARBLEState) -> str:
    """save_to_memory 이후 라우팅 결정.

    check_continue_node에서 이미 current_iteration을 증가시켰으므로,
    current_iteration <= total_iterations 이면 해당 iteration을 실행해야 함.

    Returns:
        "continue_iteration" 또는 "end"
    """
    current_iteration = state.get("current_iteration", 1)
    total_iterations = state.get("total_iterations", 1)

    # check_continue에서 이미 다음 iteration으로 증가됨
    # 증가된 값이 total 이하면 계속 진행
    if current_iteration <= total_iterations:
        return "continue_iteration"
    return "end"


def _auto_detect_stage(iteration: int) -> tuple[str, str]:
    """파일 존재 여부에 따라 자동으로 시작 stage 결정.

    가장 진행된 stage부터 역순으로 확인 (development → debate).
    해당 stage의 필수 파일이 있으면 그 stage부터 시작.
    최대 development까지만 감지 (docker는 development 이후 자동 진행).

    Args:
        iteration: 현재 iteration 번호

    Returns:
        (detected_stage, reason) 튜플
        - detected_stage: 감지된 시작 stage
        - reason: 감지 이유 (로그용)
    """
    build_path = get_build_path(iteration)

    # 1. Development stage 확인: implementation_proposal.md 존재
    proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
    if proposal_path.exists():
        return ("development", "implementation_proposal.md 발견 → development부터 시작")

    # 2. iter 2+: 이전 iteration 결과 확인
    if iteration > 1:
        prev_build_path = get_build_path(iteration - 1)
        prev_proposal = prev_build_path / "build_debate_outputs" / "implementation_proposal.md"
        if prev_build_path.exists() and prev_proposal.exists():
            return ("debate", f"build_{iteration - 1} 결과 있음 → debate부터 (iteration_critic 포함)")

    # 3. 기본값: debate
    return ("debate", "파일 없음 → debate부터 시작")


def _validate_stage_requirements(stage: str, iteration: int) -> tuple[bool, str, str]:
    """stage별 필요한 파일 존재 여부 검증.

    Args:
        stage: 검증할 stage ("debate", "development", "docker")
        iteration: 현재 iteration 번호

    Returns:
        (valid, fallback_stage, message) 튜플
        - valid: 해당 stage 진행 가능 여부
        - fallback_stage: 불가능할 경우 대체 stage
        - message: 로그 메시지
    """
    build_path = get_build_path(iteration)

    # iter 2+ debate stage: 이전 iteration 폴더 필요 (iteration_critic이 읽음)
    if stage == "debate" and iteration > 1:
        prev_build_path = get_build_path(iteration - 1)
        if not prev_build_path.exists():
            return (
                False,
                "debate",
                f"build_{iteration - 1} 폴더 없음 - 이전 iteration 결과 없음"
            )
        # 이전 iteration의 핵심 파일들 확인
        prev_proposal = prev_build_path / "build_debate_outputs" / "implementation_proposal.md"
        prev_weakness = prev_build_path / "build_debate_outputs" / "weakness_of_target_model.md"
        if not prev_proposal.exists() and not prev_weakness.exists():
            return (
                False,
                "debate",
                f"build_{iteration - 1}에 proposal/weakness 파일 없음"
            )
        return (True, stage, f"build_{iteration - 1} 이전 iteration 확인됨")

    if stage == "development":
        # development stage: implementation_proposal.md 필요
        proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
        if not proposal_path.exists():
            return (
                False,
                "debate",
                f"implementation_proposal.md 없음 → debate부터 시작"
            )
        return (True, stage, f"implementation_proposal.md 확인됨")

    elif stage == "docker":
        # docker stage: src/ 디렉토리에 .py 파일 필요
        src_path = build_path / "src"
        if not src_path.exists():
            return (
                False,
                "debate",
                f"src/ 폴더 없음 → debate부터 시작"
            )
        py_files = list(src_path.glob("*.py"))
        if not py_files:
            # src 폴더는 있지만 py 파일 없음 → development부터
            # development 가능한지도 확인
            proposal_path = build_path / "build_debate_outputs" / "implementation_proposal.md"
            if proposal_path.exists():
                return (
                    False,
                    "development",
                    f"src/에 .py 파일 없음 → development부터 시작"
                )
            else:
                return (
                    False,
                    "debate",
                    f"src/에 .py 파일 없고 proposal도 없음 → debate부터 시작"
                )
        return (True, stage, f"src/에 {len(py_files)}개 .py 파일 확인됨")

    # debate stage는 항상 가능
    return (True, "debate", "debate stage는 항상 가능")


def route_from_inject_memory(state: MARBLEState) -> str:
    """inject_memory_context 이후 stage 기반 라우팅.

    Continue 모드에서 --stage 플래그에 따라 적절한 노드로 라우팅.
    첫 iteration에서만 stage가 적용되고, 이후 iteration은 debate부터 시작.

    Stage 검증:
        - auto: 파일 존재 여부에 따라 자동 감지 (최대 development)
        - development: build_N/build_debate_outputs/implementation_proposal.md 필요
        - docker: build_N/src/*.py 파일 필요
        - 파일이 없으면 자동으로 이전 stage로 fallback

    Routes:
        - "debate" → build_debate_subgraph (기본값)
        - "development" → build_development_subgraph
        - "docker" → docker_execution_subgraph

    Returns:
        다음 노드 이름
    """
    is_continue_mode = state.get("is_continue_mode", False)
    continue_stage = state.get("continue_stage", "debate")
    current_iteration = state.get("current_iteration", 1)

    # Continue 모드가 아니면 debate부터 시작
    if not is_continue_mode:
        logger.info("[ITERATION] 일반 모드 → debate_subgraph")
        return "debate"

    # continue_stage가 None이면 (다음 iteration) debate부터
    if continue_stage is None:
        logger.info("[ITERATION] Continue 모드 (다음 iteration) → debate_subgraph")
        return "debate"

    # Auto detection: 파일 존재 여부에 따라 자동으로 stage 결정
    if continue_stage == "auto":
        detected_stage, reason = _auto_detect_stage(current_iteration)
        logger.info(f"[ITERATION] Auto detection: {reason}")
        logger.info(f"[ITERATION] Continue 모드 → {detected_stage}_subgraph (iter {current_iteration})")
        return detected_stage

    # Stage 검증 (development, docker)
    if continue_stage in ("development", "docker"):
        valid, fallback_stage, message = _validate_stage_requirements(continue_stage, current_iteration)
        if not valid:
            logger.warning(f"[ITERATION] Stage 검증 실패: {message}")
            logger.info(f"[ITERATION] Fallback → {fallback_stage}_subgraph (iter {current_iteration})")
            return fallback_stage
        else:
            logger.info(f"[ITERATION] Stage 검증 통과: {message}")
        logger.info(f"[ITERATION] Continue 모드 → {continue_stage}_subgraph (iter {current_iteration})")
        return continue_stage

    # debate (기본값)
    logger.info(f"[ITERATION] Continue 모드 → debate_subgraph (iter {current_iteration})")
    return "debate"
