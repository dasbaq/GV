# RUNBOOK.md

## 워크플로우 — M2 전처리 / Kaggle GPU 학습

라운드는 다음 환경 분업을 기본으로 한다.

| 단계 | 실행 환경 | 산출물 |
|---|---|---|
| 카탈로그 생성 | M2 Air 로컬 | `data/mock/<round>.h5` |
| sanity/floor 분석 | M2 Air 로컬 | `data/logs/*floor*.json`, `data/target_scaler_<round>.pkl` |
| ML 학습 + bootstrap | Kaggle CUDA | checkpoint, eval JSON |
| 결과 회수/문서화 | M2 Air 로컬 | `data/checkpoints/`, `data/logs/`, 문서 |

## 새 라운드 체크리스트

1. M2에서 카탈로그를 생성하고 label sanity, floor/oracle 분석을 끝낸다.
2. scaler를 `data/target_scaler_<round>.pkl`에 생성한다.
3. M2에서 equivalence phase를 실행해 CPU vs accelerator forward 및 multi-seed 1-epoch 분포를 고정한다.

```bash
python scripts/phase4_v0_1_round.py --phase equivalence --device mps --epochs 1
```

4. 생성된 `data/logs/<round>_equivalence.json`을 카탈로그/scaler와 함께 Kaggle Dataset에 포함한다.
5. Kaggle Dataset 업로드 전 dry-run을 확인한다.

```bash
python scripts/sync_to_kaggle.py --round phase4_v0_1 --init-dataset --slug lens-phase4-v0-1
```

6. 실제 업로드가 필요하면 `--execute`를 붙인다. Kaggle CLI 인증은 표준
   `~/.kaggle/kaggle.json`을 사용한다. 이 파일은 절대 커밋하지 않는다.
7. Kaggle에서 `notebooks/kaggle_round_template.ipynb`를 복사해 라운드명과 스크립트명을 맞춘다.
8. attached dataset 경로가 자동 감지되는지 확인한 뒤 2 epoch sanity를 먼저 실행한다.
   sanity는 `--phase train --epochs 2 --bootstrap-n 0`으로 실행하며 acceptance/leak trigger는
   `--min-epochs-for-acceptance` 기본값 10 때문에 `skipped_smoke`로 기록된다.
9. full run은 다음 형태로 실행한다.

```bash
python scripts/phase4_v0_1_round.py \
  --phase train \
  --equivalence-from /kaggle/input/<slug>/phase4_v0_1_equivalence.json \
  --device cuda --workers 4 --epochs 50 --bootstrap-n 1000
```

10. 결과 회수 전 dry-run을 확인한다.

```bash
python scripts/fetch_kaggle_results.py --notebook <owner/kernel-slug>
```

11. 실제 회수는 `--execute`를 붙인다. 기존 파일명 충돌 시 timestamp suffix가 붙는다.
12. 로컬에서 결과 JSON을 분석하고 `STATUS.md`, `CHANGELOG.md`, `BENCHMARKS.md`,
    `DECISIONS.md`를 필요한 범위만 갱신한다.

## 표준 환경변수

| 변수 | 의미 |
|---|---|
| `LENS_DATA_PATH` | 학습/검증 HDF5 절대경로 |
| `LENS_DATA_PATH_UNFILTERED` | selection bias 평가용 HDF5 절대경로 |
| `LENS_DATA_ROOT` | 개별 path 미지정 시 찾을 데이터 prefix |
| `LENS_WORK_ROOT` | checkpoint/log/scaler 출력 prefix |
| `LENS_SCALER_PATH` | scaler pkl 절대경로 |

M2 기본값은 repo 내부 `data/`이고, Kaggle 기본 출력은 `/kaggle/working`으로 둔다.

## 보존 규칙

- `.h5`, `.pkl`, `.pt`, `data/mock/`, `data/checkpoints/`, `data/logs/`, `data/runs/`는 git에 커밋하지 않는다.
- Kaggle Dataset이 카탈로그/scaler 공유 경로다.
- checkpoint와 eval JSON은 Kaggle output에서 회수해 로컬 ignored 디렉토리에 보관한다.
- round 스크립트에서 모델 구조, 입력 차원, loss, optimizer, batch size, AMP 정책은 라운드별 acceptance 없이 바꾸지 않는다.
