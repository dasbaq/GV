# BENCHMARKS.md
> 검증 실행 결과 요약. 실제 원본 benchmark가 없는 항목은 MOCK/SKIP로 구분.

---

## [2026-05-27] Mode 1 실제 관측 ingestion + benchmark wiring

실제 원본 관측 파일은 아직 로컬에 없어 TDC1/SDSS real benchmark 판정은 skip이다.
이번 검증은 raw 관측 ingestion 경로와 benchmark wiring의 synthetic smoke다.

| 항목 | 결과 | 판정 |
|------|------|------|
| observed ingestion unit/integration | `8 passed` | PASS |
| TDC1 real benchmark hook | artifact 없음 | SKIP |
| SDSS J1226 real benchmark hook | artifact 없음 | SKIP |
| 기존 observation/run_mode1 회귀 | `18 passed` | PASS |

검증 명령:
- `pytest -q tests/test_observed_mode1_ingestion.py tests/benchmarks/test_tdc1.py::test_tdc1_rung0_real tests/benchmarks/test_sdss_j1226.py::test_sdss_j1226_real`
  → 8 passed, 2 skipped.
- `pytest -q tests/test_observation_io.py tests/test_delay_extraction_obs.py tests/test_run_mode1_e2e.py tests/test_observed_mode1_ingestion.py`
  → 18 passed.
- `python -m py_compile ml/data_adapters/observed_mode1.py pipelines/ingest_observation.py tests/benchmarks/test_tdc1.py tests/benchmarks/test_sdss_j1226.py`
  → pass.
- `pytest -q tests/benchmarks/test_tdc1.py tests/benchmarks/test_sdss_j1226.py`
  → 4 passed, 3 skipped.

실제 파일 배치 규약:
- TDC1 Rung 0: `data/observations/tdc1_rung0_observed.h5`,
  `data/observations/tdc1_rung0_sidecar.yaml`, 선택 `tdc1_rung0_delay_config.json`.
- SDSS J1226-0006: `data/observations/sdss_j1226_observed.h5`,
  `data/observations/sdss_j1226_sidecar.yaml`, 선택 `sdss_j1226_delay_config.json`.

## [2026-05-27] Phase 4 v0.7 post-hoc sigma scaling diagnostic

v0.7은 재훈련 없이 Mode 1 predicted sigma만 `1.47x` 스케일하는 uncertainty
post-calibration으로 판정한다. 아래 v0.7 수치는 사용자 제공 진단이며, 이번 로컬 세션에서는
코드 경로와 회귀 테스트만 실행했다.

### 공식 filtered val 기준

| 항목 | 현재 | 기준/해석 | 판정 |
|------|------|-----------|------|
| filtered val n | `50` | 작은 표본, coverage 표준오차 약 `±0.066` | 주의 |
| 1σ coverage | `0.42`, CI `[0.28, 0.57]` | 목표 `[0.62, 0.78]`와 non-overlap | FAIL |
| abs_res/sigma mean | `1.47` | sigma가 실제 잔차 대비 작음 | scale source |
| QQ tail | `N=+3 -> 3.42`, `N=-3 -> -5.83` | SIE 극단 케이스 heavy-tail | 진단 |
| post-hoc sigma | `pred_sigma * 1.47` | abs_res/sigma를 약 `1.0`으로 이동 | 적용 방침 |

### unfiltered selection-bias 진단

| 항목 | 결과 | 해석 |
|------|------|------|
| 1σ coverage | `0.735` | unfiltered에서는 sigma 과대추정 경향 |
| abs_res/sigma mean | `0.74` | filtered와 반대 방향 |
| RMSE | `3.52` | filtered와 큰 격차 |
| r | `0.794` | 더 큰 집합의 진단 지표로만 사용 |

### 로컬 검증

- `python -m py_compile scripts/phase4_v0_2_round.py pipelines/run_mode1.py` → pass.
- `pytest -q tests/test_posthoc_sigma_scaling.py tests/test_run_mode1_e2e.py tests/test_heads.py` → 16 passed.
- 기존 v0.2 checkpoint/scaler로 eval-only smoke:
  `python scripts/phase4_v0_2_round.py --eval-only --device cpu --mode1-sigma-scale 1.47 --bootstrap-n 10 --workers 0`
  → filtered 1σ coverage `0.86`, CI `[0.733, 0.942]`; H0 r/RMSE는 기존 v0.2 checkpoint 특성으로 공식 v0.7 판정 아님.
- Mode 1 CLI ML correction smoke:
  synthetic observation HDF5 + `data/checkpoints/phase4_v0_2_imgres_best.pt`
  + `data/target_scaler_phase4_v0_2.pkl` + `--mode1-sigma-scale 1.47`
  → `H0_approx=69.9667`, `H0=85.7612`, `h0_correction=15.7945`,
  `sigma_H0_raw=2.3573`, `sigma_H0_scaled=3.4652`, `use_image=false`.

재평가 명령:
- `python scripts/phase4_v0_2_round.py --eval-only --mode1-sigma-scale 1.47 --bootstrap-n 1000`

## [2026-05-22] Phase 1 observation-format Δt extraction MOCK

원본 system6 실측 파일이 없어 `ObservedLensSystem.light_curves` 포맷을 흉내낸 합성
unresolved light curve로 smoke 검증했다. 실측 benchmark 판정은 MOCK/SKIP로 유지한다.

| 항목 | 결과 | 기준 | 판정 |
|------|------|------|------|
| 입력 | synthetic system6-format light curve | 실측 없음 | MOCK/SKIP |
| 주입 Δt | `24.140000 days` | — | — |
| 회복 Δt_obs | `24.050000 days` | — | — |
| Δt 오차 | `0.090000 days` | `< 0.15 days` | MOCK PASS |
| μ 추정 | `0.100000` | `abs(μ) < 1` | PASS |
| Σ minimum | `-0.800088` | configured conservative smoke threshold `-0.7` | PASS |
| confidence | `conservative` | not rejected | PASS |

검증 명령:
- `pytest -q tests/test_delay_extraction_obs.py` → 3 passed.

## [2026-05-04] Phase 3 v2.6 infra equivalence + bootstrap

### Infra Equivalence

| 항목 | 결과 | 기준 | 판정 |
|------|------|------|------|
| Forward-only max diff | max `1.04e-07` | `≤ 1e-4` | PASS |
| CPU val_m1 mean ±2std | `[0.681171, 0.878453]` | MPS mean 포함 | PASS |
| MPS val_m1 mean | `0.783326` | CPU band 내부 | PASS |
| Welch t-test | `p=0.925110` | `p > 0.05` | PASS |
| MPS speedup | `6.30x` | `≥ 2x` | PASS |
| MPS workers effect | `0.954x` | `<1.0이면 workers=0` | workers=0 선택 |

### v2.6 H0 Evaluation

| Metric | v2.5 | v2.6 |
|--------|------|------|
| best epoch | 9 | 8 |
| early stop epoch | 17 | 16 |
| best val_m1 | `0.432934` | `0.630474` |
| model RMSE | `2.532` | `3.055` |
| no-correction RMSE | `4.623` | `4.973` |
| perfect-kappa-oracle RMSE | `2.972` | `3.184` |
| model H0 r | `0.904` | `0.844` |
| 1σ coverage | `0.790` | `0.650` |

### Bootstrap 1000

| Quantity | Mean | 95% CI |
|----------|------|--------|
| gap = oracle_RMSE - model_RMSE | `0.135` | `[-0.075, 0.349]` |
| model RMSE | `3.053` | `[2.743, 3.357]` |
| oracle RMSE | `3.188` | `[2.857, 3.519]` |
| model r | `0.844` | `[0.803, 0.879]` |
| 1σ coverage resample | `0.650` | `[0.585, 0.710]` |
| 1σ coverage Clopper-Pearson | `0.650` | `[0.580, 0.716]` |

### Distribution Checks

| Normal quantile | empirical residual / predicted σ |
|-----------------|----------------------------------|
| -3 | `-2.427` |
| -2 | `-1.812` |
| -1 | `-0.979` |
| 0 | `0.039` |
| 1 | `1.082` |
| 2 | `2.262` |
| 3 | `3.295` |

Val `H0_true` KS test against `U(60,80)`: statistic `0.02258`, `p=0.25567`.

### 판정

`gap` CI가 0을 포함하고, model RMSE는 floor band `[2.7, 3.6]` 안에 있으며,
1σ coverage CI가 calibration 목표 `[0.62, 0.78]`와 겹친다.

v2.* 트랙 종료 권고: YES
