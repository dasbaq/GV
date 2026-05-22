# CHANGELOG.md
> 매 세션 마지막 5개만 읽음. 최신 항목이 위에 오도록 작성.

---

## [2026-05-22] — Phase 4 v0.2 Kaggle Dataset dry-run 준비
- `scripts/sync_to_kaggle.py`: round handoff 파일 목록에 `data/logs/phase4_v0_2_floor_analysis.json`을 포함하고 dry-run 파일 표시를 repo-relative source/staged name으로 정리.
- `python scripts/sync_to_kaggle.py --round phase4_v0_2 --init-dataset --slug lens-phase4-v0-2` dry-run 확인: train HDF5, unfiltered eval HDF5, target scaler, equivalence JSON, floor JSON 모두 포함, missing 0.
- 실제 Kaggle upload는 미실행. 사용자 승인 후 `--execute`로 진행 대기.

## [2026-05-22] — Phase 4 v0.2 floor/scaler/equivalence 확인
- `data/mock/phase4_v0_2.h5` 및 `data/mock/phase4_v0_2_eval_unfiltered.h5` 존재와 schema sign(`true_minus_approx`) 확인.
- `data/target_scaler_phase4_v0_2.pkl`: Mode 1 mean `17.4868`, scale `5.7334`로 finite/non-zero 확인.
- `data/logs/phase4_v0_2_floor_analysis.json`: `mode1_H0_correction` mean/std/min/max `17.3624/5.8614/5.9792/32.2613`, NaN/Inf 0, `|mu_truth| < 1` 확인.
- `data/logs/phase4_v0_2_equivalence.json`: 기존 MPS handoff 기준 forward max diff `1.043e-07`, Welch p `0.9981`, passed. 현재 Codex 프로세스는 MPS unavailable로 재실행 명령이 시작 전 중단됨.
- 같은 scaler로 CPU 1-epoch seed 42/1337/7 보조 점검 시 train loss `1.2928/1.3547/1.4253`, NaN batch 0.

## [2026-05-22] — Mode 1 관측 → H0 엔드투엔드 CLI 추가
- `pipelines/run_mode1.py`: 관측 HDF5 입력에서 `ObservedLensSystem` 로드 → Δt_obs 추출 → SIE fit Δφ 산출 → `invert_h0` 실행 → JSON 출력 CLI 추가.
- `pipelines/run_mode1.py`: Phase 5 ML 보정 훅을 `--apply-correction`으로 마련하되 기본 off, checkpoint 부재/ML inference 미구현 시 `correction skipped`를 명시.
- `tests/test_run_mode1_e2e.py`: forward SIE/SIS self-consistency 관측 HDF5를 생성해 CLI와 함수 경로에서 H0 회복 검증 추가.

## [2026-05-22] — 실관측 포맷 광도곡선 Δt 추출 경로 추가
- `inversion/delay_extraction.py`: `ObservedLensSystem.light_curves`에서 Phase 1 벡터화 Δt/μ grid를 적용해 `dt_obs_days`, μ, Σ 진단, MOCK 플래그를 반환하는 파이프라인 추가.
- `tests/test_delay_extraction_obs.py`: 합성 system6 포맷 MOCK에서 주입 Δt 회복 및 `|mu| < 1` guard/벡터화 검증 추가.

## [2026-05-22] — Mode 1 Δφ 단위/import 정합성 수정
- `inversion/mode1_h0.py`: Mode 1 입력 Δφ 단위를 [rad²]로 통일하고 `approx_level=0` EXACT 거리 경로를 `core.physics.distances`로 연결.
- `tests/test_mode1_consistency.py`: `standard_approx.invert_h0_from_delay_sie`와 Mode 1 EXACT 구현의 H0 일치 및 arcsec² 오입력 회귀 테스트 추가.

## [2026-05-22] — 관측 상 위치 → SIE fit → Δφ 모듈 추가
- `inversion/sie_fit.py`: 관측 상 위치에서 SIE 파라미터를 least-squares로 역피팅하고 `dphi_rad2`를 산출하는 모듈 추가.

## [2026-05-22] — 관측 입력 어댑터 추가
- `inversion/observation_io.py`: Mode 1/2용 실관측 최소 입력 dataclass와 dict/HDF5 로더 추가.

## [2026-05-11] — Phase 4 v0.2 CUDA outlier validity filter

### 변경
- `ml/data/error_catalog.py`: Phase 4 v0.2 CUDA fp16/AMP 안정화용 p99 validity filter 추가.
  기존 v0.1 기준(root/finite/`dt_true > 0`/`abs(mu_truth) < 0.98`/분리/H0/dphi)은 보존하고,
  full-truth catalog에서만 아래 신규 기준을 AND 결합한다.
  - `abs(mu_truth) <= 0.9699`
  - `dphi_sie / dphi_truth in [0.5878, 0.9201]`
  - truth image separation `>= 0.6598 arcsec`
  - `dt_approx <= 444.7 days`
  - `I_obs.sum() <= 77.79`
  - `max(abs(F_joint)) <= 3.408`
  - `abs(mode1_H0_correction) <= 32.27`
- `ml/data/error_catalog.py`: v0.2 reject reason과 threshold metadata/log 기록 추가.
  off/off sanity용 `include_nfw=False, include_kappa_ext=False` catalog에는 v0.2 p99 filter를 적용하지 않고
  v0.1 validity만 적용한다.
- `ml/data/error_catalog.py`: `--log-path`, `--reject-log-path`, `--diagnosis-log-path`,
  `--resample-budget`, `--validity-filter {v0_2,v0_1,off}`, `--eval-role` CLI 옵션 추가.
- `scripts/phase4_v0_2_round.py`: v0.2 artifact 이름을 쓰는 round 스크립트 추가.
  모델 구조, loss, optimizer, batch size, AMP 정책 및 acceptance 임계는 v0.1 round와 동일하게 유지.
- `tests/test_phase4_validity.py`: v0.2 신규 임계의 경계 통과/탈락 단위 테스트 추가.

### 산출물
- `data/mock/phase4_v0_2.h5`: n=500, seed=42.
- `data/mock/phase4_v0_2_eval_unfiltered.h5`: n=200, seed=42,
  `validity_filter="off_for_eval; root convergence required"`.
- `data/logs/phase4_v0_2_label_distribution.json`
- `data/logs/phase4_v0_2_reject_log.json`
- `data/logs/phase4_v0_2_eval_unfiltered_label_distribution.json`
- `data/logs/phase4_v0_2_eval_unfiltered_reject_log.json`
- `data/target_scaler_phase4_v0_2.pkl`: train split 기반 Mode 1/3 mean/std scaler.

### 결과
- v0.2 `mode1_H0_correction`: mean `17.362`, std `5.861`,
  min/max `5.979/32.261`.
- v0.1 대비 std `6.273 -> 5.861`, max `34.747 -> 32.261`.
- v0.2 unfiltered eval `mode1_H0_correction`: mean `30.378`, std `13.515`,
  min/max `6.543/69.339`; reject는 root convergence 계열만 발생
  (`root_find_residual=14`, `dedupe_lt2=9`).
- resample attempts mean `2.776`, max `16`.
- 신규 reject counts:
  `dphi_ratio_outside_v0_2_p01_p99=9`, `dt_approx_gt_v0_2_p99=4`,
  `image_separation_lt_v0_2_p01=6`, `image_sum_gt_v0_2_p99=5`,
  `lc_absmax_gt_v0_2_p99=4`, `mu_truth_gt_v0_2_p99=8`.

### 검증
- `python -m py_compile ml/data/error_catalog.py scripts/phase4_v0_2_round.py`
- `pytest -q tests/test_phase4_validity.py tests/test_standard_approx.py tests/test_error_catalog.py tests/test_truth_image_solver.py`
  → 12 passed.
- `python -c "import h5py; f=h5py.File('data/mock/phase4_v0_2.h5','r'); print(f['metadata'].attrs['n_systems'])"`
  → `500`.
- v0.1/v0.2 `mode1_H0_correction` std/min/max/percentile 비교 출력 확인.

## [2026-05-05] — 워크플로우 구조 정착 (M2 전처리 / Kaggle GPU 학습)

### 추가
- `scripts/lib/round_common.py`: round 스크립트 공통 환경변수, 경로 builder,
  device 선택, DataLoader kwargs, `--phase {equivalence,train,all}` CLI helper 추가.
- `scripts/sync_to_kaggle.py`: M2 산출물(`.h5`, scaler, unfiltered eval, equivalence JSON)을
  Kaggle Dataset 업로드 staging/dry-run으로 묶는 helper 추가.
- `scripts/fetch_kaggle_results.py`: Kaggle notebook output 또는 zip에서 checkpoint/log를
  로컬 `data/checkpoints/`, `data/logs/`로 회수하는 helper 추가. 기존 파일은 덮어쓰지 않는다.
- `notebooks/kaggle_round_template.ipynb`: Kaggle CUDA 실행 템플릿 추가.
- `RUNBOOK.md`: M2 equivalence → Kaggle train → M2 fetch/documentation 체크리스트 추가.

### 변경
- `scripts/v2_6_round.py`, `scripts/phase4_v0_1_round.py`:
  - `LENS_DATA_PATH`, `LENS_DATA_PATH_UNFILTERED`, `LENS_DATA_ROOT`,
    `LENS_WORK_ROOT`, `LENS_SCALER_PATH` 표준 지원.
  - `--phase equivalence`: forward diff + multi-seed 1-epoch 분포만 실행하고
    `data/logs/<round>_equivalence.json`을 출력.
  - `--phase train`: Kaggle CUDA 학습/eval만 수행하고 `--equivalence-from` JSON을 보고에 합성.
  - `--phase all`: 기존 결합 흐름 호환.
  - `--min-epochs-for-acceptance` gate 추가. 짧은 sanity는 `acceptance: skipped_smoke`,
    `leak_triggered: null`로 기록한다.
  - NaN loss 진단 필드(`nan_detected`, `nan_batches`, grad/parameter norm)를 epoch history에 기록.
- `.gitignore`: `kaggle.json`, `**/kaggle.json` ignore 추가.
- `AGENTS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `STATUS.md`: M2/Kaggle 분업과 phase 정책 반영.

### 검증
- `python -m py_compile scripts/lib/round_common.py scripts/v2_6_round.py scripts/phase4_v0_1_round.py scripts/sync_to_kaggle.py scripts/fetch_kaggle_results.py`
- `python scripts/phase4_v0_1_round.py --help`
- `python scripts/v2_6_round.py --help`
- `notebooks/kaggle_round_template.ipynb` JSON parse 확인.
- 로컬 sandbox에서 CUDA/MPS가 unavailable이라 `--phase equivalence`와 GPU train smoke는 실행하지 못함.

## [2026-05-04] — Phase 4 v0.1 truth path 정정 + outlier 정책

### 변경
- `core/physics/standard_approx.py`: SIE image 후보를 `scipy.optimize.root`로
  SIE-only 해까지 refine하도록 보강. off/off sanity가 다시 0에 수렴하도록 수정.
- `ml/data/error_catalog.py`: full truth Fermat을 SIE image 위치가 아니라
  truth lens 아래 numerical root-find한 `theta_truth`에서 평가하도록 정정.
- truth solver는 `theta - alpha_truth(theta) - beta = 0`을 풀며, 초기값은 SIE-only image.
  v0.1은 SIE-anchored search만 수행하고 truth-only extra image 전역 탐색은 하지 않는다.
- validity criteria + reject/resample 도입:
  root 수렴, finite 값, `dt_true > 0`, `abs(mu_truth) < 0.98`,
  truth image separation `>= 0.1 arcsec`, `H0_approx in [45, 90]`,
  `dphi_sie / dphi_truth in [0.5, 1.5]`.
- `LENS_RESAMPLE_BUDGET` 환경변수로 시스템당 resample budget override 지원
  (기본 50).

### 산출물
- `data/mock/phase4_v0_1.h5`: n=500.
- `data/logs/phase4_v0_1_label_distribution.json`: variance decomposition 및 v2.6 baseline 비교 포함.
- `data/logs/phase4_v0_1_reject_log.json`: reject/resample 빈도 및 root residual 기록.
- `data/logs/phase4_v0_diagnosis.json`: v0 SIE-position artifact, D1-D4 진단, E3 n_systems 정정 기록.
- `data/target_scaler_phase4_v0_1.pkl`: Phase 4 v0.1 smoke 전용 identity scaler.
- `tests/test_truth_image_solver.py`, `tests/test_phase4_validity.py` 추가.

### 결과
- v0 실제 시스템 수: 50. v0.1은 plan대로 500 생성.
- v0.1 `mode1_H0_correction`: mean `17.390`, std `6.273`,
  min `3.198`, max `34.747`.
- v2.6 baseline std `3.775` 대비 ratio `1.662`.
- variance decomposition: off/off `4.10e-19`, NFW-only `39.199`,
  kappa-only `4.195`, both `39.354`, cross term `-4.040`.
- truth image shift mean `1.030 arcsec`, max `4.765 arcsec`.
- root residual norm median `1.72e-12`, max `1.79e-08`.
- resample attempts mean `2.614`, max `12`; reject counts:
  `H0_approx_outside_45_90=670`, `root_find_residual=65`,
  `mu_truth_ge_0p98=40`, `dedupe_lt2=30`, `image_separation_lt_0p1=2`.
- F1 dedupe threshold `0.01 arcsec`; dedupe된 accepted roots는 0회.
- ParamEncoder 입력 13차원은 변경 없음; D3 truth-leak 없음.

### 검증
- `python -m py_compile core/physics/standard_approx.py ml/data/error_catalog.py`
- `pytest -q tests/test_truth_image_solver.py tests/test_phase4_validity.py tests/test_standard_approx.py tests/test_error_catalog.py`
  → 10 passed.
- Phase 4 v0.1 HDF5 + `target_scaler_phase4_v0_1.pkl`로
  `LensCorrectionDataset` 및 `MultiModalErrorCorrector` forward+backward 1 batch smoke 통과.
- 전체 `pytest -q tests/ --tb=short`: 85 passed, 6 skipped, 19 failed.
  실패는 D4 분류대로 legacy ML fixture/schema cleanup 대상이며 Phase 4 v0.1 category C는 0개.

## [2026-05-04] — Phase 4 standard_approx + error_catalog 1차 구현

### 추가
- `core/physics/standard_approx.py`: SIE 단일 표준 근사 적용 함수와 closed-form
  Mode 1 H0 역산 helper 추가. 입력은 공개 SIE approximation 키만 허용하고
  `M200`, `concentration`, `kappa_ext`, `nfw_offset` truth-only 키는 명시적으로 거부.
- `ml/data/error_catalog.py`: full truth와 SIE approx를 페어링해
  `true_values/`, `approx_outputs/`, `correction_targets/` HDF5 그룹을 생성.
- `tests/test_standard_approx.py`, `tests/test_error_catalog.py`: signature/docstring,
  truth-only 키 차단, HDF5 schema, `correction = true - approx`, off/off sanity 검증 추가.
- `data/mock/phase4_v0.h5`: n=50 small 검증 카탈로그 생성.
- `data/logs/phase4_v0_label_distribution.json`: B1 variance decomposition 및
  B2 v2.6 baseline std read-only 비교 기록.
- `data/target_scaler_phase4_v0.pkl`: Phase 4 smoke 전용 identity scaler 생성.

### 결정 반영
- truth physics: `alpha_truth(theta) = alpha_SIE(theta) + alpha_NFW(theta) + kappa_ext * theta`.
  NFW는 origin-aligned, `kappa_ext` 범위는 `[0, 0.1]`로 유지.
- 부호 정정: Phase 4부터 correction label은 HDF5 schema 정의대로
  `true - approx`. v2.*의 `approx - true` 계열 scaler/checkpoint와 호환하지 않는다.
- Mode 2 correction은 정식 solver 전까지 zeros 유지.
- Mode 3 correction은 source-plane `S_true - S_approx`만 저장한다.
- Phase 4 v0만 `image_size=64`, `pixel_scale=0.1 arcsec/pix`로 시야를 유지한다.
  Phase 4 v1부터 기본 128 복귀 예정.

### 검증
- `python -m py_compile core/physics/standard_approx.py ml/data/error_catalog.py`
- `pytest -q tests/test_standard_approx.py tests/test_error_catalog.py` → 6 passed.
- `python -m ml.data.error_catalog --n-systems 50 --output data/mock/phase4_v0.h5`
- Phase 4 HDF5 + `target_scaler_phase4_v0.pkl`로 `LensCorrectionDataset` 및
  `MultiModalErrorCorrector` forward+backward 1 batch smoke 통과.
- 전체 `pytest -q tests/`: 81 passed, 6 skipped, 19 failed. 실패는 기존 ML 테스트 fixture의
  stale param/image shape 및 legacy mock HDF5 schema 문제로 확인되어 다음 ML 정리 라운드에서 처리.

## [2026-05-04] — GitHub push 전 대용량 산출물 ignore 보강

### 추가
- `.gitignore`: `*.h5`, `*.pkl`, `*.pt`, `data/mock/`, `data/checkpoints/`,
  `data/logs/`, `data/runs/`, `data/cached_dataset.pt`, `data/test_samples/` ignore 추가.
- pytest/IDE 로컬 캐시(`.pytest_cache/`, `.vscode/`, `.idea/`) ignore 추가.

### 검증
- `git status --short`: `data/mock/`, `data/checkpoints/`, `data/logs/`,
  `data/runs/`, `*.pkl`, `*.pt`, `*.h5`가 untracked 목록에서 사라짐.
- `git status --ignored --short data`: 해당 산출물이 ignored(`!!`)로 표시됨.

## [2026-05-04] — v2.6 Kaggle 이식 최소 패치

### 변경
- `scripts/v2_6_round.py`: `LENS_DATA_ROOT`, `LENS_DATA_PATH`, `LENS_WORK_ROOT`
  환경변수를 지원해 Kaggle read-only input과 `/kaggle/working` output을 분리.
- `scripts/v2_6_round.py`: `--device {auto,cuda,mps,cpu}`, `--workers`,
  `--worker-candidate`, `--epochs` CLI 옵션 추가.
- `scripts/v2_6_round.py`: CPU vs accelerator forward/distribution/speed 검증을 MPS 고정에서
  CUDA/MPS 공용으로 일반화. CUDA에서는 기본 worker 후보를 4로 두고, `--workers`가 주어지면
  full retrain/eval worker 수를 명시값으로 고정.
- `scripts/v2_6_round.py`: repo 밖 출력 경로(`/kaggle/working`)도 JSON에 안전하게 기록하도록
  path 표시를 보강.
- `scripts/v2_6_round.py`: smoke 실행용 `--bootstrap-n 0`에서 빈 bootstrap percentile 계산으로
  실패하지 않도록 0회 bootstrap CI를 `null`로 기록.
- `requirements.txt`: `torch>=2.1.0`로 명시 버전 보강.

### 검증
- `python -m py_compile scripts/v2_6_round.py`
- `python scripts/v2_6_round.py --help`
- env var 미설정 시 기본 경로가 repo 내부 `data/mock/real_phase3_v2_6.h5` 및
  `data/{target_scaler,checkpoints,logs}`로 유지됨을 import sanity로 확인.

## [2026-05-04] — Phase 3 v2.6 인프라 동등성 재정의 + MPS full retrain

### 추가
- `scripts/v2_6_round.py`: CPU/MPS forward-only op 검증, 다중 seed 1-epoch 분포 검증,
  CPU/MPS/MPS+workers 속도 측정, 부분 산출물 삭제, v2.6 재학습, bootstrap 평가를 일괄 실행.

### 산출물
- `data/mock/real_phase3_v2_6.h5`는 보존하고 재사용.
- 삭제 후 재생성:
  `data/target_scaler_phase3_v2_6.pkl`,
  `data/checkpoints/phase3_v2_6_imgres_best.pt`.
- 신규/갱신 로그:
  `data/logs/phase3_v2_6_infra_equivalence.json`,
  `data/logs/phase3_v2_6_imgres_long_history.json`,
  `data/logs/phase3_v2_6_imgres_h0_eval.json`.

### 검증
- `python -m py_compile scripts/v2_6_round.py`
- Forward-only CPU/MPS max abs diff: `mode1_pred=2.24e-08`,
  `mode2_pred=4.47e-08`, `mode3_pred=1.04e-07`,
  `log_sigma_mode1=7.45e-09`, `log_sigma_mode2=4.47e-08` → `1e-4` 기준 통과.
- 다중 seed 1 epoch `val_m1`: CPU mean `0.779812 ± 0.049320`,
  MPS mean `0.783326`; CPU mean ±2std band `[0.681171, 0.878453]` 안에 포함,
  Welch `p=0.925110` → 통과.
- 속도: CPU `782.48s`, MPS workers=0 `124.25s` (`6.30x`),
  MPS workers=4 `130.30s` (`6.01x`). workers=4가 느려 full retrain은 workers=0 선택.
- v2.6 training: best epoch 8, early stop epoch 16, best `val_m1=0.630474`.
- v2.6 H0 eval: model RMSE `3.055`, no-correction RMSE `4.973`,
  perfect-kappa-oracle RMSE `3.184`, model H0 r `0.844`.
- Bootstrap 1000회: gap(oracle-model RMSE) mean `0.135`, 95% CI `[-0.075, 0.349]`;
  model RMSE 95% CI `[2.743, 3.357]`, model r 95% CI `[0.803, 0.879]`.
- Calibration: 1σ coverage `65.0%`, Clopper-Pearson 95% CI `[0.580, 0.716]`;
  2σ coverage `95.0%`, `|z|>3` rate `0.5%`.
- 분포 점검: val H0_true KS vs `U(60,80)` statistic `0.02258`, `p=0.25567`.
- Acceptance 판정: gap CI가 0을 포함하고 model RMSE/coverage CI가 목표 band와 겹쳐
  v2.* 트랙 종료 및 Phase 4 진입 권고.

## [2026-05-04] — Phase 3 v2.5 light-curve/Σ truth leak 차단 + analytic floor 재정의

### 변경
- `src_py/simulation/generator.py`: light curve에 임베딩되는 delay를 truth-side
  `delta_t_obs = dt_true * (1-kappa_ext)`에서 `dt_approx_noisy`로 변경.
- `_sigma_curve`의 peak 중심을 `dt_true`에서 `dt_approx_noisy`로 변경하고,
  `dt_lc_sigma = dt_approx_noisy * dt_sigma_rel` 기반 width를 사용하도록 수정.
- `generator_version=phase3-v2.5`, `metadata/lc_delay_source=dt_approx_noisy` 추가.
- ParamEncoder 입력은 v2.4의 8개 입력 그대로 유지.

### 산출물
- `data/mock/real_phase3_v2_5.h5`
- `data/target_scaler_phase3_v2_5.pkl`
- `data/checkpoints/phase3_v2_5_imgres_best.pt`
- `data/logs/phase3_v2_5_imgres_long_history.json`
- `data/logs/phase3_v2_5_imgres_h0_eval.json`

### 검증
- 사전 analytic floor(v2.3/v2.5 분포 동일): 전체 `H0_mean * mean(sigma_rel)=3.168`,
  `linear H0*epsilon RMSE=3.310`, perfect-kappa-oracle RMSE `3.305`,
  val perfect-kappa-oracle RMSE `2.972`.
- `python -m py_compile src_py/simulation/generator.py ml/training/dataset.py`
- `python src_py/simulation/generator.py --n_systems 500 --output data/mock/real_phase3_v2_5.h5 --seed 42`
- v2.5 training: best epoch 9, early stop epoch 17, best `val_m1=0.432934`.
- v2.5 H0 eval: model RMSE `2.532`, no-correction RMSE `4.623`,
  perfect-kappa-oracle RMSE `2.972`, perfect-joint-oracle RMSE `0.000004`,
  model H0 r `0.904`.
- Calibration: 1σ coverage `79%`, 2σ coverage `98%`, `|z|>2` rate `2%`, `|z|>3` rate `0%`.
- Acceptance 판정: H0 r 통과, model은 perfect-kappa-oracle보다 개선됨.
  RMSE는 목표 band `[2.7, 3.6]`보다 약간 낮고 1σ coverage는 목표 상단 0.78보다 0.01 높아 borderline.

## [2026-05-03] — Phase 3 v2.4 ParamEncoder truth-side leak 차단

### 변경
- `ml/training/dataset.py`: ParamEncoder 입력에서 truth-side `params/H0`, `dt_ratio`,
  `params/M200`, `params/concentration` 제거.
- ParamEncoder 입력을 inference-side 값으로 재정의:
  `H0_approx`, `z_lens`, `z_source`, `sigma_v`, `q`, `theta_E`, `dt_lc`, `dt_lc_sigma`.
- `dt_lc = approx_outputs/dt_approx`, `dt_lc_sigma = 0.045 * dt_lc`로 정의해
  `perturbations/delta_t_obs`, `kappa_ext`, `dt_sigma_rel`, `dt_approx_noise_factor`가
  ParamEncoder로 흐르지 않도록 차단.
- `config/ml.yaml`의 `param_normalization`을 v2.4 입력 차원에 맞게 갱신.

### 산출물
- 데이터는 `data/mock/real_phase3_v2_3.h5` 재사용.
- `data/target_scaler_phase3_v2_4.pkl`
- `data/checkpoints/phase3_v2_4_imgres_best.pt`
- `data/logs/phase3_v2_4_imgres_long_history.json`
- `data/logs/phase3_v2_4_imgres_h0_eval.json`

### 검증
- `python -m py_compile ml/training/dataset.py src_py/simulation/generator.py`
- v2.4 training: best epoch 8, early stop epoch 16, best `val_m1=0.407288`.
- v2.4 H0 eval: model RMSE `2.4573`, no-correction RMSE `4.6231`,
  perfect-kappa-oracle RMSE `2.9718`, perfect-joint-oracle RMSE `0.000004`.
- v2.4 model H0 r `0.9071`; 수용 기준 0.95~0.99보다 낮아 입력이 다소 빈약한 상태로 판단.
- log_sigma calibration: 1σ coverage `84%`, 2σ coverage `98%`, outlier >3σ `0%`;
  v2.3의 100% coverage에서는 내려왔지만 목표 0.55~0.80보다 약간 높음.

## [2026-05-03] — Phase 3 v2.3 dt_approx 노이즈 주입 + H0 재평가

### 변경
- `src_py/simulation/generator.py`: `dt_approx`를 `dt_true`와 동일하게 저장하던 v2.2 oracle-like 경로를 제거하고,
  시스템별 `dt_sigma_rel ~ U(0.02, 0.07)` 및 Gaussian relative noise를 적용.
- `approx_outputs/dt_approx`, `approx_outputs/H0_approx`, `correction_targets/mode1_H0_correction`,
  `simplification_errors/mode1_H0_error`를 noisy `dt_approx` 기반 라벨로 일관되게 재정의.
- `perturbations/dt_sigma_rel`, `perturbations/dt_approx_noise_factor` 및 metadata
  `dt_noise_model=gaussian_relative_uniform_0p02_0p07` 추가.
- `generator_version`을 `phase3-v2.3`으로 갱신.

### 산출물
- `data/mock/real_phase3_v2_3.h5`
- `data/target_scaler_phase3_v2_3.pkl`
- `data/checkpoints/phase3_v2_3_imgres_best.pt`
- `data/logs/phase3_v2_3_imgres_long_history.json`
- `data/logs/phase3_v2_3_imgres_h0_eval.json`

### 검증
- `python -m py_compile src_py/simulation/generator.py`
- `python src_py/simulation/generator.py --n_systems 500 --output data/mock/real_phase3_v2_3.h5 --seed 42`
- v2.3 sanity: `corr(mode1_H0_error, kappa_ext)=-0.5328`,
  `corr(mode1_H0_error, dt_approx_noise_factor)=-0.8224`,
  `partial_corr(mode1_H0_error, dt_noise | kappa_ext)=-0.9878`.
- v2.3 best training: best epoch 22, early stop epoch 30, `val_m1=0.002741`.
- v2.3 H0 eval: model RMSE `0.2044`, no-correction RMSE `4.6231`,
  perfect-kappa-oracle RMSE `2.9718`, model H0 r `0.9995`.
- log_sigma calibration은 over-conservative: `|residual| <= 1σ` coverage `100%`.

## [2026-05-03] — Prototype smoke 통과 + Phase 3 quasar_lc 구현

### 추가
- `src_py/simulation/__init__.py`
- `src_py/simulation/quasar_lc.py`: DRW/CARMA AR(1) quasar light curve 생성기와 ZTF/LSST/ideal 노이즈 모델.

### 변경
- `src_py/ml/multimodal_dataset.py`: 사용되지 않는 `core.lightcurve.LightCurveSimulator` import와 이를 위한
  `sys.path` 설정 제거.
- `src_py/ml/train.py`: macOS 안정성을 위해 `num_workers=0`으로 변경하고 `pin_memory`는 CUDA에서만 활성화.
- `STATUS.md`: Phase 3 `quasar_lc.py`를 완료로 표시.
- 프로덕션 ML 파이프라인 mock smoke train+backward 검증 완료.
  Phase 3 시뮬레이션 코어 착수: quasar_lc.py / noise_model.py.
- Phase 3 generator.py 1차 구현: SIELens + quasar_lc + noise_model을 엮어 production HDF5 schema 호환
  시뮬레이션 50개 생성, LensCorrectionDataset/MultiModalErrorCorrector forward+backward 통과.
  image, simplification_errors는 image_renderer.py 및 Phase 4에서 교체 예정.
- Phase 3 image_renderer.py 구현 (inverse ray tracing + Gaussian PSF).
  generator.py에 Mass-Sheet Degeneracy 기반 κ_ext 섭동 주입 →
  mode1_H0_error 라벨이 유의미한 분산을 갖도록 수정.
  image/source residual은 image_renderer 결과 사용.
  Mode 2 DM 라벨은 여전히 zeros (Phase 4에서 처리).

### 검증
- `cd src_py/ml && python -u train.py`: 50 epochs 완료, `gravitational_lens_model.pth` 저장.
- `python src_py/ml/generate_mock_dataset.py --n_samples 500 --seed 42`
- `LensCorrectionDataset(..., split="train", max_len=200, sigma_curve_size=50, image_size=64)` HDF5 로드: `PASS`
- `LensCorrectionDataset(..., param_norm=config/ml.yaml)` params shape `[11]`: `PASS`
- `python src_py/simulation/quasar_lc.py`: `PASS`
- `python src_py/simulation/image_renderer.py`: `PASS`
- `python src_py/simulation/generator.py --n_systems 50 --output data/mock/real_phase3_v1.h5 --seed 42`
- `data/mock/real_phase3_v1.h5`: `mode1_H0_error std=2.129`, image std=0.076, `LABELS NON-TRIVIAL OK`
- `MultiModalErrorCorrector` forward+backward on `real_phase3_v1.h5`: loss `2.8878`, `PASS`

## [2026-05-03] — Prototype ML 버그 수정 + DRW mock HDF5 생성

### 추가
- `src_py/ml/generate_mock_dataset.py`: DRW/CARMA AR(1) light curve, SIE lens parameter,
  mock image/source, approx output, correction target을 포함한 HDF5 생성기.
- `data/mock/mock_dataset.h5`: 10,000개 synthetic mock lens system.

### 변경
- `src_py/ml/train.py`: cwd와 무관하게 `data/cached_dataset.pt`를 찾도록 `__file__` 기반 경로로 수정.
- `src_py/ml/train.py`: 근거 없는 hyperparameter 변경을 이전값
  (`batch_size=64`, `epochs=50`, `lr=0.001`)으로 되돌림.
- `src_py/ml/inference.py`: MC dropout 추론 시 Dropout만 train mode로 켜고 BatchNorm은 eval mode 유지.
- `DECISIONS.md`: Phase 3/4 부재를 메우는 DRW 기반 mock 데이터 사용 결정을 기록.

### 검증
- `python src_py/ml/generate_mock_dataset.py --n_samples 24 --seed 42 --out /private/tmp/mock_dataset_smoke.h5`
- `python src_py/ml/generate_mock_dataset.py --n_samples 10000 --seed 42`
- `python -c "from pathlib import Path; from ml.training.dataset import LensCorrectionDataset; ds=LensCorrectionDataset([Path('data/mock/mock_dataset.h5')], split='train', max_len=200, sigma_curve_size=50, image_size=64); print(len(ds))"` → `48000`
- `python -c "from pathlib import Path; from src_py.ml.generate_mock_dataset import verify_dataset; verify_dataset(Path('data/mock/mock_dataset.h5')); print('verify ok')"`
- `python -m py_compile src_py/ml/train.py src_py/ml/inference.py src_py/ml/generate_mock_dataset.py`

## [2026-05-02] — Phase 2 중력렌즈 물리 모델 구현

### 추가
- `core/physics/{config,distances,refractive_index,lens_models,ray_tracing}.py`
- `core/physics/__init__.py`
- `tests/test_physics_{refractive_index,lens_models,ray_tracing,distances}.py`

### 변경
- `config/physics.yaml`에 기존 flat key를 보존하면서 `constants`, `cosmology`, `numerics` nested 구조 추가.
- Phase 2 중심 계산 흐름을 `Φ → n_eff → ∇n_eff → ray tracing → OPL → travel time → path time delay`로 구현.

### 성능
- Phase 2 단위 테스트: 16 passed, 0.43초.
- 전체 테스트: 89 passed, 6 skipped, 10.59초.

### 검증
- `pytest -q tests/test_physics_refractive_index.py tests/test_physics_lens_models.py tests/test_physics_ray_tracing.py tests/test_physics_distances.py`: 16 passed.
- `pytest -q tests/`: 89 passed, 6 skipped.

## [2026-05-02] — Phase 1 시간 지연 추출 엔진 구현

### 추가
- `core/light_curve/{io,reconstruction,smoothing,fluctuation,time_delay}.py`
- `pipelines/extract_td.py`
- `tests/{test_reconstruction,test_smoothing,test_fluctuation,test_time_delay,test_io}.py`
- `tests/benchmarks/_mock_data.py`
- `data/mock/system6_synthetic.h5`

### 변경
- Phase 1 `core/` 패키지 도입으로 legacy `src_py` 모듈이 기대하던 `core.base_sim` import를
  호환 shim으로 재노출.

### 성능
- 단일 시스템 Phase 1 smoke test: 전체 Phase 1 단위 테스트 9개 0.49초.
- 전체 테스트: 73 passed, 6 skipped, 10.17초.

### 검증
- `pytest -q tests/`: 73 passed, 6 skipped.
- 실제 benchmark 원본 데이터(system6, ZTF, SDSS J1226, TDC1 Rung 0/1)는 저장소에 없어 MOCK/SKIP 상태 유지.

## [2026-05-02] — 다중 근사 프로파일 폐기 → 단일 표준 근사 (SIE) 채택

### 변경
- 4개 근사 프로파일(`FULL_NUMERICAL` / `SIE_LENS` / `SIS_LENS` / `POINT_LENS`) 체계 폐기.
  프로젝트 전체에서 **단 하나의 표준 근사**(SIE 타원 + 단일 평면 + 등방 + κ_ext=0 + 평활 mass)만 사용.
- ML이 어떤 단순화가 적용됐는지 알 필요가 없어짐 → **axes one-hot 입력 제거**.
- 인코더·데이터셋·역산 솔버에서 `approximation_profile` 인자 모두 제거.
  표준 근사는 코드 구현에 implicit하게 포함되며, 외부에서 토글 불가.

### 추가
- ARCHITECTURE.md §"표준 근사 모델 (Standard Approximation)" 섹션 신설.
  표준 근사의 구체적 내용(6개 항목)과 SIE를 선택한 근거 명시.
- HDF5 스키마에 `approx_outputs/`, `correction_targets/` 그룹 신설.

### 수정
- HDF5 `params/approximation/` 그룹 제거 (시스템별 axes 정보 더 이상 불필요).
- HDF5 `metadata/default_profile` 제거 → `metadata/full_truth_available`만 유지.
- `simplification_errors/` → `correction_targets/`로 명칭 변경 (의미 명확화).
- ML 입력 ParamEncoder: 물리 파라미터만 (one-hot 차원 제거).
- DECISIONS.md의 "axes_options 사전 고정" 결정을 **superseded**로 마킹.

### 성능
- (해당 없음 — 설계 변경만)
- 예상 효과: 학습 데이터의 분포가 단일 오차 패턴으로 좁아져 ML 수렴 속도·품질 개선 기대.

### 검증
- (해당 없음 — 코드 미실행)

---

## [2026-05-02] — 근사 모델을 단순화 프로파일로 재정립 + Bag et al. ε 공식 2D화

### 변경
- **근사 레벨(EXACT/FAST/TURBO 0/1/2) 개념 폐기**.
  실제 의도는 "수치 정확도 단계"가 아니라 **물리적 parametric 단순화**였음.
- 6개의 **단순화 축**으로 재정의 *(이후 [2026-05-02 두 번째 항목]에서 단일 근사로 다시 통합)*.
- 4개의 대표 프로파일 신설 *(이후 단일 SIE 표준 근사로 통합)*.
- Bag et al. 2022 시간 지연 공식을 (Δt_try, μ_try) **2D 그리드** 스윕으로 명시:
  - ε(Δt_try) → ε(Δt_try, μ_try)
  - Σ(Δt_try) → Σ(Δt_try, μ_try)

### 추가
- ML 모델 ③번 모달리티(Σ 곡선)을 **2D CNN**으로 격상 (Δt × μ 평면)

### 수정
- HDF5 `metadata/approx_level` (int) → `metadata/full_truth_available` (bool, 단일 근사 채택 후)
- 렌즈 모델 표 갱신

---

## [2026-04-30] — Mode 1/2/3 정의 재정립

### 변경
- Mode 정의를 "근사 레벨 매핑"에서 **세 가지 역산 task**로 재정의:
  - Mode 1: 허블 상수 H₀ 역산
  - Mode 2: 암흑물질 질량 분포 역산
  - Mode 3: 아인슈타인 링/다중상 이미지에서 source 복원 (구현 완료, 외부 모듈)
- ARCHITECTURE.md §"세 가지 역산 Mode" 신설, ML 모델 구조를 공유 인코더 +
  Mode별 분기 헤드 구조로 변경
- HDF5 스키마에 `images/` 그룹과 Mode 1·2·3별 ground-truth/error 필드 추가
- STATUS.md Phase 5를 5-A(역산 솔버) / 5-B(ML 보정)로 분리

### 추가
- 4번째 입력 모달리티: 관측 이미지 I_obs (Mode 3 활성 시)
- 검증 현황 표에 Mode 2·3 벤치마크 자리 표시

### 수정
- 근사 레벨(0/1/2)과 Mode 번호는 **직교 축**임을 명시.
  *(2026-05-02 다시 변경: 근사 레벨 → 다중 프로파일 → 단일 표준 근사)*

---

<!-- 아래 형식을 복사해서 위에 추가하세요 -->
<!--
## [YYYY-MM-DD] — 한 줄 요약

### 추가
-

### 변경
-

### 수정
-

### 성능
-

### 검증
-
-->
