"""
IC50 Comparison Visualization - Multi-Model
6개 DRP 모델의 약물별 IC50 분포를 3x2 (3행 2열)로 비교
각 모델: 실험값(GDSC2=y_true) vs 예측값(model=y_pred)
"""

import json
import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from utils.plot_style import setup_plot_style, AUTODRP_COLORS, get_model_color
from utils.filename_helper import get_timestamped_filename
from utils.atc_parser import add_atc_levels_to_dataframe

logger = logging.getLogger(__name__)


class IC50ComparisonVisualizationTool:
    """6개 DRP 모델의 약물별 IC50 분포를 3x2 grouped boxplot으로 비교하는 도구"""

    def __init__(self):
        setup_plot_style()
        # model_order will be set dynamically from input
        self.model_order: list[str] = []
        # 실험값과 예측값을 구분하기 위한 색상
        self.colors = {
            'gdsc2': AUTODRP_COLORS['primary'],  # 실험값: 파란색
            'model': AUTODRP_COLORS['accent']  # 예측값: 오렌지색
        }

    def _prepare_drug_data(self, df: pd.DataFrame, drug_list: list, data_type: str, group_col: str = 'DRUG_NAME'):
        """
        약물별로 데이터 준비 (boxplot용)

        Args:
            df: 모델 데이터프레임 (DRUG_NAME, y_true, y_pred 컬럼)
            drug_list: 표시할 약물 리스트
            data_type: 'y_true' (GDSC2) 또는 'y_pred' (모델 예측)
            group_col: 그룹화할 컬럼명 (DRUG_NAME 또는 ATC_LEVEL_*)

        Returns:
            약물별 데이터 리스트
        """
        box_data = []
        for drug in drug_list:
            drug_data = df[df[group_col] == drug]
            values = drug_data[data_type].dropna().values
            if len(values) > 0:
                box_data.append(values)
            else:
                box_data.append([])  # 빈 리스트 추가
        return box_data

    def apply(self,
              model_csv_paths: Dict[str, str],
              top_n_drugs: Optional[int] = None,
              atc_level: Optional[int] = None,
              output_dir: str = "figures") -> str:
        """
        6개 모델의 약물별 IC50 분포를 3x2 grouped boxplot으로 시각화

        각 subplot: GDSC2(실험값, y_true) vs 모델(예측값, y_pred)

        Args:
            model_csv_paths: 6개 모델 CSV 파일 경로 딕셔너리
                            예: {"deepdr": "path/to/deepdr.csv", ...}
                            CSV 필수 컬럼: CELL_LINE_NAME, DRUG_NAME, y_true, y_pred
            top_n_drugs: 상위 N개 약물만 표시 (None이면 모든 약물 표시)
            atc_level: ATC code 계층 레벨 (1-5). None이면 개별 약물(DRUG_NAME) 사용
                      1=Anatomical, 2=Therapeutic, 3=Pharmacological, 4=Chemical, 5=Substance
            output_dir: PNG 파일 저장 디렉토리 (기본값: "figures")

        Returns:
            JSON 문자열 (status, output_path, statistics 포함)
        """
        try:
            # 입력 검증: 6개 모델 필수
            if len(model_csv_paths) != 6:
                return json.dumps({
                    "status": "error",
                    "message": f"Expected 6 models, got {len(model_csv_paths)}"
                })

            # Set model_order dynamically from input
            self.model_order = list(model_csv_paths.keys())

            # 모든 모델 데이터 로드
            model_data = {}
            all_drugs_set = set()

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

                # Add ATC level column if needed
                if atc_level is not None:
                    df = add_atc_levels_to_dataframe(df, atc_column='ATC_CODE', levels=[atc_level])

                model_data[model_name] = df

                # Determine grouping column (drug name or ATC level)
                if atc_level is not None:
                    group_col = f'ATC_LEVEL_{atc_level}'
                else:
                    group_col = 'DRUG_NAME'

                all_drugs_set.update(df[group_col].dropna().unique())

            # 공통 약물 리스트 생성 (모든 모델에 존재하는 약물 우선)
            # 각 약물의 총 샘플 수 계산
            drug_sample_counts = {}
            for drug in all_drugs_set:
                total_count = sum(
                    len(df[df[group_col] == drug])
                    for df in model_data.values()
                )
                drug_sample_counts[drug] = total_count

            # 샘플 수 기준 정렬
            sorted_drugs = sorted(drug_sample_counts.items(),
                                 key=lambda x: x[1], reverse=True)

            # top_n_drugs 적용
            if top_n_drugs is not None and top_n_drugs > 0:
                drug_list = [drug for drug, _ in sorted_drugs[:top_n_drugs]]
            else:
                drug_list = [drug for drug, _ in sorted_drugs]

            n_drugs = len(drug_list)
            if n_drugs == 0:
                return json.dumps({
                    "status": "error",
                    "message": "No drugs found in the datasets"
                })

            # 3x2 subplot 생성 (3행 2열)
            fig, axes = plt.subplots(3, 2, figsize=(20, 24))
            axes = axes.flatten()

            all_statistics = {}

            # 각 모델별로 subplot 생성
            for idx, model_name in enumerate(self.model_order):
                if model_name not in model_data:
                    return json.dumps({
                        "status": "error",
                        "message": f"Model '{model_name}' not found in input"
                    })

                df = model_data[model_name]
                ax = axes[idx]

                # GDSC2(y_true)와 모델(y_pred) 데이터 준비
                gdsc2_data = self._prepare_drug_data(df, drug_list, 'y_true', group_col)
                model_pred_data = self._prepare_drug_data(df, drug_list, 'y_pred', group_col)

                # Boxplot 위치 계산
                positions_gdsc2 = []
                positions_model = []

                for i in range(n_drugs):
                    base_pos = i * 3  # 약물 간 간격
                    positions_gdsc2.append(base_pos - 0.5)
                    positions_model.append(base_pos + 0.5)

                # GDSC2 boxplot (파란색)
                bp1 = ax.boxplot(
                    gdsc2_data,
                    positions=positions_gdsc2,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=self.colors['gdsc2'], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=self.colors['gdsc2'], linewidth=1.5),
                    capprops=dict(color=self.colors['gdsc2'], linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                   markerfacecolor=self.colors['gdsc2'],
                                   markeredgecolor='none')
                )

                # 모델 예측 boxplot (오렌지색)
                bp2 = ax.boxplot(
                    model_pred_data,
                    positions=positions_model,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=self.colors['model'], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=self.colors['model'], linewidth=1.5),
                    capprops=dict(color=self.colors['model'], linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                   markerfacecolor=self.colors['model'],
                                   markeredgecolor='none')
                )

                # X축 설정
                ax.set_xticks([i * 3 for i in range(n_drugs)])
                ax.set_xticklabels(drug_list, rotation=45, ha='right', fontsize=10)

                # 축 레이블
                ax.set_xlabel('Drug Name', fontsize=12, fontweight='bold')
                ax.set_ylabel('LN(IC50)', fontsize=12, fontweight='bold')

                # Subplot 제목
                ax.set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold', pad=10)

                # 범례 (첫 번째 subplot에만)
                if idx == 0:
                    legend_elements = [
                        Patch(facecolor=self.colors['gdsc2'], alpha=0.7,
                              edgecolor='black', label='GDSC2 (Experimental)'),
                        Patch(facecolor=self.colors['model'], alpha=0.7,
                              edgecolor='black', label='Model (Predicted)')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right',
                             fontsize=10, frameon=True, fancybox=True, shadow=True)

                # 그리드
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                # 통계 정보 계산
                model_stats = {}
                for drug in drug_list:
                    drug_df = df[df[group_col] == drug]

                    gdsc2_values = drug_df['y_true'].dropna()
                    model_values = drug_df['y_pred'].dropna()

                    model_stats[drug] = {
                        'gdsc2': {
                            'n': len(gdsc2_values),
                            'median': float(gdsc2_values.median()) if len(gdsc2_values) > 0 else None,
                            'mean': float(gdsc2_values.mean()) if len(gdsc2_values) > 0 else None
                        },
                        'model_pred': {
                            'n': len(model_values),
                            'median': float(model_values.median()) if len(model_values) > 0 else None,
                            'mean': float(model_values.mean()) if len(model_values) > 0 else None
                        }
                    }

                all_statistics[model_name] = model_stats

            # 전체 타이틀
            if atc_level is not None:
                title = f'IC50 Distribution Comparison by ATC Code (Level {atc_level}) - 6 Models'
            else:
                title = 'IC50 Distribution Comparison by Drug - 6 Models'
            if top_n_drugs is not None:
                title += f' (Top {top_n_drugs})'
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Figure 저장
            os.makedirs(output_dir, exist_ok=True)
            filename = get_timestamped_filename("ic50_comparison_6models", "png")
            output_path = os.path.join(output_dir, filename)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return json.dumps({
                "status": "success",
                "output_path": output_path,
                "statistics": all_statistics,
                "metadata": {
                    "n_models": 6,
                    "models": self.model_order,
                    "n_drugs": n_drugs,
                    "drug_list": drug_list,
                    "top_n_drugs": top_n_drugs,
                    "atc_level": atc_level,
                    "grouping": "ATC Level" if atc_level else "Individual Drug",
                    "output_file": filename
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"IC50 comparison visualization error: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
