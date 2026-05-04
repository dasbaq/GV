# BENCHMARKS.md
> 검증 실행 결과 요약. 실제 원본 benchmark가 없는 항목은 MOCK/SKIP로 구분.

---

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
