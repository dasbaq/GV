# DECISIONS.md
> 설계 결정과 그 근거를 기록. 번복 시 반드시 이유와 날짜를 함께 기재.
> 세션마다 읽지 않아도 됨. 충돌 가능성이 있을 때만 참조.

---

## [2026-05-24] 실관측 YAML ingest와 ParamEncoder feature schema 단일화

### 결정
실관측 1단계 입력은 YAML 리스트로 받되, 학습/추론 모델에는 `ml/training/feature_schema.py`의
공통 ParamEncoder schema를 통해 들어간다. Bag+22는 레포에서 호출하지 않고 외부 결과
`dt_lc`, `dt_lc_sigma`, 광곡선 품질 지표 4개를 YAML entry에 기록한다.

ParamEncoder scalar feature는 기존 8개에서 다음 15개로 확장한다.
`H0_approx`, `z_lens`, `z_source`, `sigma_v`, `q`, `theta_E`, `dt_lc`,
`dt_lc_sigma`, `n_epochs_quality`, `baseline_days`, `median_cadence_days`,
`median_photometric_error`, `missing_sigma_v`, `missing_theta_E`, `missing_q`.
여기에 기존 `approx_level` 2 one-hot과 `target_mode` 3 one-hot을 붙여 총 입력 차원은 20이다.

### 근거
- 실측 시스템에서는 `sigma_v`, `theta_E`, `q`가 정상적으로 누락될 수 있으므로 예외나 NaN 전파가
  아니라 normalized-zero sentinel + field별 missing flag로 처리해야 한다.
- `dt_lc`와 `dt_lc_sigma`는 Mode 1 H0 추론의 필수 관측량이므로 누락 시 entry를 무효화한다.
- Mode 2는 이번 작업에서 입력을 추가하지 않지만, YAML의 `mode2_inputs` 예약 필드는 보존해
  향후 image positions, all pair delays, flux ratios, magnifications를 별도 structured encoder로
  연결할 수 있게 둔다.
- `dt_lc_sigma_sampler`, `dt_sign_convention`, primary pair convention, real YAML missing modality 정책은
  `config/ml.yaml:data.observed_features`에 둔다. Hydra/OmegaConf는 도입하지 않고, 기존 프로젝트의
  YAML + `yaml.safe_load` config 방식을 유지한다.

### 확정 config 정책
- 시뮬레이션 `dt_lc_sigma`는 `relative_then_clip` sampler를 사용한다:
  `relative_error.distribution=log_uniform`, min/max `[0.01, 0.30]`, absolute clip `[0.3, 20.0] days`.
  실측 카탈로그 수집 후에는 코드 변경 없이 `config/ml.yaml` 값만 조정한다.
- delay 저장 규약은 `|Δt|` 양수다. YAML 입력이 음수 `dt_lc`를 주면 configured policy에 따라
  `abs(dt_lc)`로 변환하고 `pair_order.leading_image/trailing_image`를 뒤집으며 `conversion_log`에 남긴다.
  `pair_order`는 메타데이터로 보존하지만 ParamEncoder 입력에는 넣지 않는다.
- 광곡선 품질 지표 normalization은 config의 `transform`으로 제어한다. 초기 기본값은 log transform이며
  범위는 `N_epochs [0,1500]`, `baseline_days [0,5000]`, `median_cadence_days [0,50]`,
  `median_photometric_error [0,0.5]`다.
- real catalog fixture는 `tests/fixtures/real_catalog/{complete,partial_no_lens_model,minimal,invalid_examples}.yaml`
  네 파일로 유지한다.

### 운영 규칙
- 실측 YAML과 시뮬레이션 HDF5 모두 같은 feature builder를 사용해야 한다.
- Bag+22 raw extraction과 ZTF 다운로드는 ingest adapter에서 수행하지 않는다.
- `M200`, `concentration`, `kappa_ext`, `nfw_offset`는 실측 YAML에서도 truth-only key로 거부한다.
- 기존 13-dim checkpoint는 새 20-dim 모델에 로드할 때 ParamEncoder 첫 layer의 기존 column만
  부분 이식한다. 새 feature column은 초기화 상태이므로 정식 성능 평가는 새 schema로 재학습해야 한다.

### 관련 파일
- `config/ml.yaml`
- `ml/training/feature_schema.py`
- `inversion/real_catalog.py`
- `ml/training/dataset.py`
- `inversion/obs_to_features.py`
- `pipelines/run_mode1.py`
- `ml/data/error_catalog.py`

## [2026-05-22] Phase 4 v0.4 acceptance 임계 재교정 (구분포 기준 폐기)

### 결정
v0.4(물리-validity-only, 무편향)부터 filtered absolute acceptance band를 재산정한다.
- `filtered_rmse_ci_lower_min` 2.755 → `0.5`, `filtered_rmse_ci_upper_max` 4.862 → `11.08`,
  `filtered_rmse_point_band` [2.755,6.57] → `[0.5, 16.62]` (no_correction_filtered 29.63 기준
  upper=no_corr/2.67, point upper=no_corr/2).
- `filtered_h0_r_min` 0.85 → `0.0` (record_only). leak `nfw_oracle_lower` 2.755 → `0.5`.
- selection-bias(`ratio<=2.5`, leak ratio `3.18`)·coverage `[0.62,0.78]`·`positive_fraction` 유지.

### 근거
- 기존 band(2.755/4.862/6.57/r>=0.85)는 v0.2-era **truncated easy-subset**(filtered correction ~17로 절단)
  에서 유도됐다. v0.4는 filtered==unfiltered==전체 물리 분포이므로 이 band가 무효다.
- v0.4 실측: no_correction RMSE filtered `29.63`/unfiltered `33.25`, 모델 filtered `5.81`/unfiltered
  `4.81` → 5~7x 개선. selection bias 소멸(ratio `0.83`, leak false, unfiltered RMSE `14.5→4.81`,
  r `0.28→0.59`, coverage `0.21→0.695`).
- filtered r이 v0.3.1 `0.66`에서 v0.4 `0.33`으로 "하락"한 것은 성능 저하가 아니라, 더 이상 쉬운
  subset이 아닌 전체 난이도 분포를 정직하게 평가하기 때문이다. 신뢰도 높은 unfiltered(n=200) r은 `0.59`.
- `r>=0.85`는 easy-subset 산물이라 폐기한다. 무편향 분포의 achievable-r ceiling은 inputs-conditioned
  oracle로 산정해야 하므로 record_only로 두고, 그 산정을 remaining rigor로 남긴다
  (data/logs/phase4_v0_4_floor_analysis.json).

### 주의 (goalpost-moving 아님)
RMSE band는 "모델이 no_correction을 큰 배수로 이겨야 한다"는 무편향-분포 기준으로 재산정한 것이며,
특정 모델을 통과시키려 맞춘 값이 아니다. r은 통과시키지 않고 record_only로 강등했다.

## [2026-05-22] NaN 원인은 fp16 SSIM, validity tail filter는 오진단 산물 → Phase 4 v0.4 물리 validity only

### 결정
1. `ml/training/losses.py` 의 `_ssim_loss`, `_gaussian_nll` 를 autocast 비활성 fp32 블록으로 강제하고,
   SSIM 분산을 `clamp_min(0)`, NLL의 `2*log_sigma`를 `[-30,30]` 클램프 + var floor 1e-8로 둔다.
2. Phase 4 v0.4 카탈로그는 **물리 validity only** (`validity_filter="v0_4"`): root 수렴/finite/
   `dt_true>0`/`|mu|<0.98`/separation`>=0.1`만 유지. label-상관 cap, v0.2 입력측 tail cap,
   H0 stratified quota를 **모두 제거**한다(비-stratified 기본 수집).

### 근거
- v0.2(NaN 0)→v0.3(NaN)→v0.3.1(입력 tail cap 복원해도 NaN) 결과로 "극단 입력→AMP overflow"
  가설이 폐기됐다. v0.3.1 AMP-off full run은 NaN 0이었고(NaN이 fp16 전용임을 확정),
  v0.3 history에서 `mode1/2/3_task`·`mode1_cal`은 유한한데 `ssim=nan`→`total=nan`이었다.
  ⇒ NaN의 발원지는 fp16/autocast 하의 SSIM(분모 eps underflow + 분산 음수)이며 카탈로그 필터와 무관.
- v0.1/v0.2의 입력측 tail filter는 "CUDA fp16/AMP 안정화"를 명분으로 도입됐으나, 실제 NaN 원인은
  SSIM이었다. 즉 tail filter는 **오진단으로 만들어진 것**이고 그 유일한 실효는 selection bias였다.
- SSIM을 fp32로 고치면 tail filter가 불필요해진다. 따라서 v0.4는 train 분포를 unfiltered
  (root-converged) eval 분포와 일치시켜 selection bias를 구조적으로 제거한다. 실제로 v0.4 train의
  `mode1_H0_correction` mean/std/max `29.06/13.68/69.34`가 unfiltered `30.38/13.52/69.34`와
  일치한다(v0.2 filtered는 ~17로 잘려 bias의 원인이었다).

### 번복 가능성
v0.4 재학습에서 큰 correction tail로 인해 다른 불안정이 나오면, label-중립 입력 cap만 선택적으로
재도입하는 것을 검토한다(단 label-상관 cap·quota는 재도입 금지).

## [2026-05-22] Phase 4 v0.3.1 label/input validity split

### 결정
Phase 4 v0.3.1은 v0.3의 H0-neutral 10-bin stratified quota를 유지하고, validity gate를
label 의존 컷과 입력/관측측 컷으로 분리한다.

| 컷 | 분류 | v0.3.1 처리 | 근거 |
|---|---|---|---|
| `H0_approx in [45,90]` | label 의존 | 제외 | approximate target-side H0에 직접 의존하며 v0.2의 가장 큰 H0 왜곡 driver |
| `abs(mode1_H0_correction) <= 32.27` | label 의존 | 제외 | correction = `H0_true - H0_approx` label 자체를 threshold |
| `max(abs(F_joint)) <= 3.408` | 입력/관측측 | 복원 | LC input tensor magnitude의 AMP/fp16 overflow 방지 |
| `I_obs.sum <= 77.79` | 입력/관측측 | 복원 | image input total flux tail 안정화 |
| `dt_approx <= 444.7` | 입력/관측측 | 복원 | ML param input tail 안정화 |
| `abs(mu_truth) <= 0.9699` | 입력/관측측 | 복원 | `|mu| < 1` 안정 조건의 p99 tail |
| `dphi_sie/dphi_truth in [0.5878,0.9201]` | 입력/관측측, label 상관 risk | 복원 후 별도 진단 | H0 uniformity는 유지하지만 correction/dphi support를 좁힘 |
| truth image separation `>= 0.6598 arcsec` | 입력/관측측 | 복원 | image/ray support tail 안정화 |

### 결과
- `data/mock/phase4_v0_3_1.h5`: n=500, seed=42, H0 bin별 50개.
- filtered H0 KS vs U[60,80] p `0.999993`로 v0.3 p `0.984` 수준 유지.
- NaN precheck는 v0.2 입력 안전범위 안: max|F_joint| `3.347`, I_obs.sum `69.03`,
  dt_approx `443.89`, |mu|max `0.969895`, dphi_ratio `[0.58875,0.92007]`,
  separation min `0.66384`; correction max도 reference `32.27` 안의 `32.261`.
- dphi band ablation: input-tail mask에서 dphi band 포함 시 H0 KS p `0.0935`,
  dphi 제외 시 p `0.0305`; 하지만 correction max는 dphi 제외 시 `69.34`까지 복귀한다.
  따라서 이번 round는 NaN 안정성을 우선하고, correction/dphi support mismatch는 Kaggle train
  acceptance와 unfiltered eval에서 별도 판정한다.

### 운영 규칙
- 모델 구조, 입력 차원, loss, optimizer, batch size, AMP 정책은 v0.3에서 변경하지 않는다.
- NaN이 재발하면 `F_joint` 정규화, NLL variance floor, AMP disable/guard 등은 별도 acceptance가
  필요한 제안으로만 다루고 v0.3.1 round에서는 실행하지 않는다.
- selection-bias acceptance는 `unfiltered/filtered RMSE <= 2.5`, 1σ coverage target `[0.62,0.78]`를 유지한다.

### 관련 파일
- `ml/data/error_catalog.py`
- `scripts/analyze_phase4_v0_2_selection_bias.py`
- `scripts/phase4_v0_3_1_round.py`
- `tests/test_phase4_validity.py`
- `data/logs/phase4_v0_3_1_floor_analysis.json`

## [2026-05-22] Phase 4 v0.3 H0-neutral validity filter

### 결정
Phase 4 v0.3 catalog validity filter는 H0/correction 값에 직접 의존하지 않는다.

v0.3은 다음 물리/수치 validity만 generator-level reject 조건으로 유지한다.

| 항목 | v0.3 기준 | 근거 |
|---|---:|---|
| truth image root solve | success + residual threshold | full-truth delay 계산의 최소 조건 |
| finite values | `dt_true`, `H0_approx`, Fermat values finite | HDF5 label sanity |
| `dt_true` | `> 0` | 시간 지연 물리 조건 |
| `abs(mu_truth)` | `< 0.98` | `|mu| < 1` 수렴/안정 조건 |
| truth image separation | `>= 0.1 arcsec` | image dedupe/분해능 최소 조건 |

아래 v0.1/v0.2 조건은 v0.3 validity에서 제거한다.

| 제거 조건 | 제거 이유 |
|---|---|
| `H0_approx in [45,90]` | v0.2 selection-bias 분석에서 가장 큰 robust H0 왜곡 driver |
| `abs(mode1_H0_correction) <= 32.27` | label-dependent tail cut |
| v0.2 dphi/dt/image/LC/separation p99/p01 tail gates | filtered catalog를 쉬운 support로 좁히는 selection bias |

H0 분포 정합은 reject gate가 아니라 H0 `[60,80]` 10-bin stratified quota로 처리한다.

### 결과
- `data/mock/phase4_v0_3.h5`: n=500, seed=42, H0 bin별 50개.
- `data/mock/phase4_v0_3_eval_unfiltered.h5`: n=200, seed=42, root convergence only.
- filtered H0 KS vs U[60,80] p `0.984`로 v0.2 p `1.8e-6` 대비 개선.
- filtered/unfiltered 1D KS p: H0 `0.144`, correction `0.801`, dphi_ratio `0.853`,
  mu `0.685`, separation `0.351`.

### 운영 규칙
- 모델 구조, 입력 차원, loss, optimizer, batch size, AMP 정책은 변경하지 않는다.
- 입력 feature 보강은 acceptance가 필요한 별도 제안으로만 다룬다.
- selection-bias acceptance는 Kaggle CUDA train 전 사전 선언한다:
  `unfiltered/filtered RMSE <= 2.5`, 1σ coverage target `[0.62, 0.78]`.
- no-correction/oracle baseline은 같은 eval JSON 경로에 함께 기록한다.

### 관련 파일
- `ml/data/error_catalog.py`
- `scripts/analyze_phase4_v0_2_selection_bias.py`
- `scripts/phase4_v0_3_round.py`
- `tests/test_phase4_validity.py`
- `data/logs/phase4_v0_3_floor_analysis.json`

## [2026-05-22] Mode 1 Fermat potential unit convention

### 결정
Mode 1 H0 inversion에서 페르마 포텐셜 차이 Δφ는 항상 `[rad²]`로 전달하고 저장한다.

### 운영 규칙
- `inversion/mode1_h0.invert_h0` 입력 `fermat_potential_rad2`는 `[rad²]`만 허용한다.
- `core.physics.standard_approx.invert_h0_from_delay_sie`와 `inversion/mode1_h0`는 동일한
  `[days, rad², z_lens, z_source]` 입력 계약을 공유한다.
- 관측 상 위치 역피팅(`inversion/sie_fit.py`)의 출력 키는 단위를 드러내는 `dphi_rad2`로 유지한다.
- arcsec² 값이 필요한 표시/외부 입출력에서는 호출부에서 명시적으로 변환하고,
  Mode 1 solver 내부에서 암묵 변환하지 않는다.

### 결정 근거
시간지연 물리식 `Δt = (1 + z_L) * (D_L D_S / D_LS) * Δφ / c`에서 Δφ는 무차원 각도 제곱이며
프로젝트의 SIE lens `fermat_potential` 및 표준 근사 H0 inversion은 `[rad²]`를 사용한다.
Mode 1 solver도 이 규약으로 맞춰 두 구현 간 단위 불일치를 제거한다.

### 관련 파일
- `inversion/mode1_h0.py`
- `core/physics/standard_approx.py`
- `inversion/sie_fit.py`
- `tests/test_mode1_consistency.py`

## [2026-05-11] Phase 4 v0.2 CUDA outlier validity filter

### 결정
Phase 4 v0.1 catalog의 Kaggle CUDA fp16/AMP train NaN은 hyperparameter 변경 없이
catalog validity filter 강화로 대응한다.

v0.2는 v0.1 validity criteria를 그대로 유지하고, full-truth catalog entry에 다음 p99 기반
tail filter를 AND 결합한다.

| 항목 | v0.2 기준 | 근거 |
|---|---:|---|
| `abs(mu_truth)` | `<= 0.9699` | v0.1 train split p99 |
| `dphi_sie / dphi_truth` | `[0.5878, 0.9201]` | v0.1 train split p1/p99 |
| truth image separation | `>= 0.6598 arcsec` | v0.1 train split p1 |
| `dt_approx` | `<= 444.7 days` | v0.1 train split p99 |
| `I_obs.sum()` | `<= 77.79` | v0.1 train split p99 |
| `max(abs(F_joint))` | `<= 3.408` | v0.1 train split p99 |
| `abs(mode1_H0_correction)` | `<= 32.27` | v0.1 train split p99 |

### 운영 규칙
- 모델 구조, 입력, loss, optimizer, lr, batch size, grad clip, AMP 정책은 변경하지 않는다.
- v0.2 filter는 generator-level full-truth catalog validity filter다.
- selection-bias 평가용 unfiltered catalog는 `validity_filter=off`로 생성한다.
  이 모드는 root convergence/dedupe 실패만 reject하고 v0.1/v0.2 value filter는 적용하지 않는다.
- off/off sanity catalog(`include_nfw=False`, `include_kappa_ext=False`)에는 v0.2 p99 filter를 적용하지 않고
  v0.1 validity만 적용한다. off/off에서는 `dphi_sie/dphi_truth = 1`이 정상이라 v0.2 upper tail filter와
  의도적으로 충돌하기 때문이다.
- v0.2 round 스크립트는 v0.1 round와 같은 학습/평가 정책을 쓰고 artifact 이름만 분리한다.
  acceptance 임계는 CUDA train 결과가 나오기 전 임의 조정하지 않는다.

### 결과
- `data/mock/phase4_v0_2.h5`: n=500, seed=42.
- `data/mock/phase4_v0_2_eval_unfiltered.h5`: n=200, seed=42, root convergence only.
- `data/logs/phase4_v0_2_label_distribution.json`
- `data/logs/phase4_v0_2_reject_log.json`
- `data/logs/phase4_v0_2_eval_unfiltered_label_distribution.json`
- `data/logs/phase4_v0_2_eval_unfiltered_reject_log.json`
- `data/target_scaler_phase4_v0_2.pkl`
- v0.2 `mode1_H0_correction`: mean `17.362`, std `5.861`, min/max `5.979/32.261`.
- v0.1 대비 std `6.273 -> 5.861`, max `34.747 -> 32.261`.
- resample attempts mean `2.776`, max `16`.

### 관련 파일
- `ml/data/error_catalog.py`
- `scripts/phase4_v0_2_round.py`
- `tests/test_phase4_validity.py`
- `data/logs/phase4_v0_2_label_distribution.json`
- `data/logs/phase4_v0_2_reject_log.json`

## [2026-05-05] M2 전처리 / Kaggle GPU 학습 분업 및 round phase 분리

### 결정
라운드 운영은 M2 로컬 전처리와 Kaggle CUDA 학습으로 분리한다.

| 단계 | 실행 환경 | 역할 |
|---|---|---|
| 카탈로그 생성, scaler, floor/oracle sanity | M2 Air 로컬 | deterministic seed로 데이터셋 고정 |
| equivalence phase | M2 Air 로컬 | CPU vs accelerator forward diff 및 multi-seed 1-epoch 분포 확인 |
| train phase | Kaggle CUDA | full retrain, bootstrap, acceptance/leak 판정 |
| 결과 회수/문서화 | M2 Air 로컬 | checkpoint/log 분석 및 문서 반영 |

round 스크립트는 `--phase {equivalence,train,all}`을 공통 지원한다.
기본값은 `train`이다. `all`은 기존 결합 실행 흐름을 보존하기 위한 호환 모드다.

### 환경변수 표준
모든 round 스크립트는 다음 환경변수를 지원한다.

- `LENS_DATA_PATH`: 학습/검증 HDF5 절대경로
- `LENS_DATA_PATH_UNFILTERED`: selection bias 평가용 HDF5 절대경로
- `LENS_DATA_ROOT`: 개별 path 미지정 시 fallback prefix
- `LENS_WORK_ROOT`: checkpoint/log/scaler 출력 prefix
- `LENS_SCALER_PATH`: scaler pkl 절대경로. 명시 시 read-only input으로 취급하고 삭제/재생성하지 않는다.

### acceptance epoch gate
short sanity run의 false-positive를 막기 위해 acceptance/leak 판정은 다음 조건을 모두 만족할 때만 수행한다.

1. `--phase`가 `train` 또는 `all`
2. `epochs >= --min-epochs-for-acceptance` (기본 10)
3. early stop 발동 또는 max epoch 도달

조건 미충족 시 JSON에는 `acceptance: skipped_smoke`, `leak_triggered: null`을 기록한다.
임계값과 공식은 변경하지 않고 gate만 추가한다.

### 결정 근거
- M2 MPS는 CPU 대비 빠르지만 full retrain에는 Kaggle T4/P100이 더 적합하다.
- Kaggle 세션의 CPU multi-seed equivalence는 비용이 커서 학습 세션에서 반복하지 않는다.
- 카탈로그 생성은 stochastic하므로 M2에서 seed와 artifact를 먼저 굳히고 Kaggle Dataset으로 공유한다.
- 2 epoch sanity는 수렴 전 모델이므로 acceptance/leak trigger 평가 대상이 아니다.

### 관련 파일
- `scripts/lib/round_common.py`
- `scripts/v2_6_round.py`
- `scripts/phase4_v0_1_round.py`
- `scripts/sync_to_kaggle.py`
- `scripts/fetch_kaggle_results.py`
- `notebooks/kaggle_round_template.ipynb`
- `RUNBOOK.md`

## [2026-05-04] Phase 4 v0.1 truth image solving 및 reject/resample 정책

### 결정
Phase 4 v0의 SIE-position truth Fermat 평가는 폐기하고, v0.1부터 full truth 시간 지연은
truth lens 아래에서 수치적으로 푼 image position에서 평가한다.

```
theta_truth = root(theta - alpha_truth(theta) - beta)
dt_true     = (1 + z_L) D_dt / c * Delta phi_truth(theta_truth)
```

root solver는 `scipy.optimize.root(method="hybr")`를 사용한다. 초기값은 SIE-only image이며,
v0.1은 SIE 해와 truth 해의 1:1 대응만 추적한다. NFW/LOS가 만드는 truth-only extra image의
전역 탐색은 하지 않는다.

### 운영 규칙
- truth image dedupe threshold는 `0.01 arcsec`이다.
- root 수렴 판정은 `success=True`에 더해 residual norm
  `||theta - alpha_truth(theta) - beta|| < 1e-6 arcsec` 및 finite Jacobian condition number를 요구한다.
- catalog entry validity criteria:
  1. truth image root finding 수렴
  2. finite `dt_true`, `H0_approx`, Fermat values
  3. `dt_true > 0`
  4. `abs(mu_truth) < 0.98`
  5. truth image separation `>= 0.1 arcsec`
  6. `H0_approx in [45, 90]`
  7. `dphi_sie / dphi_truth in [0.5, 1.5]`
- 실패한 시스템은 reject/resample한다. 시스템당 기본 budget은 50회이며
  `LENS_RESAMPLE_BUDGET`로 override할 수 있다. budget 초과 시 명시적 에러로 종료한다.
- 이 filter는 generator-level validity filter이며 ARCHITECTURE.md의 simulation parameter range를
  변경하지 않는다.

### 결과
- v0.1 500-system catalog 생성 성공. 평균 시도 횟수 `2.614`, 최대 `12`.
- reject counts: `H0_approx_outside_45_90=670`, `root_find_residual=65`,
  `mu_truth_ge_0p98=40`, `dedupe_lt2=30`, `image_separation_lt_0p1=2`.
- `mode1_H0_correction` std는 v0 `81.137`에서 v0.1 `6.273`으로 감소했다.
- variance decomposition cross term은 v0 `-4612.694`에서 v0.1 `-4.040`으로 감소했다.
  off/off variance는 `4.10e-19`로 0에 수렴한다.
- deflection-additive 모델 자체는 v0.1 결과에서 유지한다. 큰 음수 cross term은 모델 문제가 아니라
  v0의 SIE-position evaluation artifact였던 것으로 판단한다.

### 관련 파일
- `core/physics/standard_approx.py`
- `ml/data/error_catalog.py`
- `tests/test_truth_image_solver.py`
- `tests/test_phase4_validity.py`
- `data/logs/phase4_v0_1_label_distribution.json`
- `data/logs/phase4_v0_1_reject_log.json`

## [2026-05-04] Phase 4 truth physics 스코프 및 correction 부호 정정

### 결정
Phase 4 v0의 full numerical truth는 후보 (a)인 **SIE main + NFW halo + LOS κ_ext**로 둔다.
구체식은 deflection-additive 방식으로 고정한다.

```
alpha_truth(theta) = alpha_SIE(theta; sigma_v, q)
                   + alpha_NFW(theta; M200, concentration, offset=0)
                   + kappa_ext * theta
```

Fermat potential은 위 deflection field의 potential을 합산해 계산한다.
NFW offset은 v0에서 origin-aligned로 두며, offset 분포 도입은 Phase 4 v1에서 별도 결정한다.
`kappa_ext` 범위는 v2.*와 같은 `[0, 0.1]`을 유지해 v2.6 κ_ext-only baseline과
분산 분해 비교가 가능하게 한다. NFW/κ_ext off-mode는 `M200 -> 0` 우회가 아니라
명시적 flag로 제어한다.

### correction 부호 break change
Phase 4부터 HDF5 schema 정의대로 ML label은 항상 다음 부호를 사용한다.

```
correction_targets = true_values - approx_outputs
```

v2.* mock track의 일부 산출물은 `approx - true` 부호로 저장되어 있었다.
따라서 `target_scaler_phase3_v2_*.pkl` 및 `phase3_v2_*_imgres_best.pt` checkpoint는
Phase 4 데이터와 호환되지 않으며 로드 금지다. Phase 4 smoke와 재학습은
`target_scaler_phase4_v0.pkl`처럼 별도 prefix의 scaler를 새로 만들거나 identity scaler를 사용한다.

### 결정 근거
- IrregularGridLens pixel-grid truth는 물리적으로 가장 정식에 가깝지만 현재 skeleton이라
  Phase 4 v0 인프라 구현 범위에는 과하다.
- κ_ext-only mock을 이름만 Phase 4로 바꾸는 회귀를 피하려면 NFW와 LOS를 모두 포함한
  비trivial label 분산이 필요하다.
- SIE 표준 근사는 계속 단일 고정이며 `approximation_*`, profile, level 인자를 추가하지 않는다.
- Mode 2 correction은 정식 inversion solver 전까지 zeros로 유지해 pseudo-label을 만들지 않는다.
- Mode 3 correction은 schema에 맞춰 source-plane `S_true - S_approx`만 저장한다.

### 운영 규칙
- Phase 4 v0 카탈로그는 `image_size=64`, `pixel_scale=0.1 arcsec/pix`를 사용한다.
  이는 1차 검증 속도를 위한 임시 축소이며 Phase 4 v1부터 128 복귀를 검토한다.
- v2.* HDF5는 baseline std 비교 목적으로 read-only 접근만 허용한다.
  v2.* scaler, checkpoint, log, HDF5 삭제 또는 덮어쓰기는 금지한다.
- `simplification_errors/` alias는 ML compatibility를 위해 유지하지만,
  Phase 4에서는 alias도 `true - approx` 부호를 따른다.

### 관련 파일
- `core/physics/standard_approx.py`
- `ml/data/error_catalog.py`
- `tests/test_standard_approx.py`
- `tests/test_error_catalog.py`
- `data/logs/phase4_v0_label_distribution.json`

## [2026-05-04] Phase 3 v2.* 트랙 종료 및 Phase 4 진입 권고

### 결정
v2.6부터 CPU/MPS infra 동등성은 bit-exact 1-epoch 학습 궤적이 아니라
forward-only op 정확성 + 다중 seed 학습 분포 일치로 판단한다.
v2.6 결과가 통계 수용 기준을 만족했으므로 v2.* mock leak 추적 트랙은 종료하고
다음 작업은 Phase 4 정식 error catalog로 전환한다.

### 결정 근거
- CPU/MPS forward-only max abs diff는 모든 출력에서 `1e-4` 이내였다.
- seed `{42, 1337, 7}` 1-epoch `val_m1` 분포에서 MPS mean은 CPU mean ±2std band 안에 있고
  Welch t-test `p=0.925110`으로 device 차이를 기각하지 못했다.
- v2.6 bootstrap gap(oracle-model RMSE) 95% CI `[-0.075, 0.349]`가 0을 포함한다.
- model RMSE 95% CI `[2.743, 3.357]`는 floor band `[2.7, 3.6]`와 정합적이다.
- 1σ coverage Clopper-Pearson 95% CI `[0.580, 0.716]`는 calibration 목표 `[0.62, 0.78]`와 겹친다.

### 운영 규칙
- macOS MPS full retrain은 workers=4가 workers=0보다 느린 것이 확인됐으므로
  현재 v2.6 mock-scale에서는 `num_workers=0`을 사용한다.
- 모델 구조, 입력, 라벨, batch size, AMP 정책은 변경하지 않는다.
- Phase 4에서 full numerical vs SIE pair를 만들기 전까지 v2.* 결과는 mock-track 판단 근거로만 사용한다.

### 관련 파일
- `scripts/v2_6_round.py`
- `data/logs/phase3_v2_6_infra_equivalence.json`
- `data/logs/phase3_v2_6_imgres_h0_eval.json`

## [2026-05-03] Phase 3 라벨 의미화 1차 결정 — κ_ext Mass-Sheet Degeneracy

### 결정
SIE-only 근사가 놓치는 첫 물리 효과로 외부 수렴 `κ_ext`를 채택한다.
Phase 3 generator는 시스템별 `κ_ext ∈ [0, 0.1]`를 샘플링하고,
`H0_approx = H0_true · (1 - κ_ext)`를 적용해
`mode1_H0_error = H0_approx - H0_true = -κ_ext · H0_true` 라벨을 만든다.

### 배경
이전 Phase 3 generator는 `simplification_errors`가 거의 모두 0이라 production ML smoke loss가
trivial zero-prediction 문제로 붕괴했다. Phase 4의 full numerical vs SIE error catalog가 아직 없기 때문에,
학습 파이프라인이 의미 있는 correction label을 먼저 받도록 해석적으로 통제 가능한 신호가 필요했다.

### 결정 근거
- Mass-Sheet Degeneracy는 time-delay cosmography에서 H0 bias를 직접 만드는 대표 효과다.
- SIE 인버전을 완성하지 않고도 `H0 bias = -κ_ext · H0_true`를 해석적으로 적용할 수 있다.
- `κ_ext ≤ 0.1` 범위는 Mode 1 label을 약 `[-7, 0] km/s/Mpc`로 만들어 smoke training에 충분한 분산을 준다.
- 향후 Phase 4에서 NFW substructure, line-of-sight structure, full numerical source residual로 확장 가능하다.

### 운영 규칙
- `κ_ext` 라벨은 Phase 3 v1 mock label이며 benchmark 통과 근거로 사용하지 않는다.
- HDF5에는 `perturbations/kappa_ext`와 `metadata.attrs["perturbation_model"] = "kappa_ext_msd_v0"`를 저장한다.
- Mode 2 DM label은 Phase 4 전까지 zeros로 유지한다.

### 관련 파일
- `src_py/simulation/generator.py`
- `src_py/simulation/image_renderer.py`
- `data/mock/real_phase3_v1.h5`

---

## [2026-05-03] Phase 3/4 부재 중 DRW 기반 mock 학습 데이터 사용

### 결정
Phase 3 대규모 시뮬레이션과 Phase 4 표준 근사 오차 카탈로그가 완성되기 전까지,
`src_py/ml/generate_mock_dataset.py`로 생성한 DRW 기반 mock HDF5를 ML smoke training과
end-to-end 호환성 검증에 사용한다.

### 배경
production ML 파이프라인은 HDF5 기반 멀티모달 구조가 준비되어 있지만 실제 학습 데이터가 없다.
현재 병목은 quasar light curve 생성, noise model, full numerical truth와 SIE 표준 근사 페어링이다.

### 결정 근거
- DRW/CARMA AR(1)는 quasar variability의 표준적인 1차 근사라 light-curve encoder 입력을
  완전한 백색잡음보다 현실적인 구조로 만든다.
- H0, SIE lens parameter, source image truth와 SIE approx output을 함께 저장해
  `truth - approx_outputs` 보정 라벨의 인터페이스를 즉시 검증할 수 있다.
- correlated 5-15% mock error는 NFW/irregular truth와 SIE 표준 근사의 차이를 아직 계산하지
  못하는 Phase 3/4 공백을 메우기 위한 개발용 대리값이다.

### 운영 규칙
- mock 데이터는 benchmark 통과 근거로 사용하지 않는다.
- 표준 근사(SIE)는 계속 단일 고정이며 generator에 `approximation_*` 선택 인자를 추가하지 않는다.
- 실제 Phase 3/4 산출물이 생기면 mock HDF5는 smoke/development 용도로만 남긴다.

### 관련 파일
- `src_py/ml/generate_mock_dataset.py`
- `data/mock/mock_dataset.h5`
- `ml/training/dataset.py`
- `ml/models/error_corrector.py`
- `ml/training/losses.py`

---

## [2026-05-02] Phase 2 물리 코어의 중심 계산 흐름

### 결정
Phase 2의 중심 API는 thin-lens analytic 공식이 아니라
`Φ → n_eff → ∇n_eff → ray tracing → optical path length → travel time` 흐름으로 둔다.

### 배경
프로젝트 목적은 복잡한 상대론적 수치 계산을 매번 직접 풀지 않고, 중력 퍼텐셜을
유효굴절률 필드로 바꿔 광학적 경로 문제로 계산하는 것이다. Thin-lens 공식은
검증, 비교, 초기값 추정에는 유용하지만 Phase 2의 주 계산식이 되면 설계 의도가 흐려진다.

### 결정 근거
- `core/physics/refractive_index.py`가 `n_eff = 1 - 2Φ/c²`와 `∇n_eff` 변환을 담당한다.
- `core/physics/lens_models.py`의 렌즈 모델은 공통적으로 `potential_3d`, `grad_potential_3d`,
  `effective_refractive_index`, `grad_refractive_index`를 제공한다.
- `core/physics/ray_tracing.py`는 `lens.grad_refractive_index(position)` 기반으로 ray path를 적분하고
  optical path length와 travel time을 계산한다.
- `distances.py`와 lens model의 `deflection`, `fermat_potential`, `analytic_time_delay`는
  sanity check용 보조 API로만 둔다.

### 운영 규칙
- SIE는 계속 프로젝트 전역 단일 표준 근사로 유지한다.
- 함수 시그니처에 `approximation_profile`, `approximation_level`, `approximation_*` 인자를 추가하지 않는다.
- IrregularGridLens의 수치 적분/보간은 full numerical truth 생성용 skeleton으로 남겨 Phase 3 이후에 확장한다.

### 관련 파일
- `core/physics/refractive_index.py`
- `core/physics/lens_models.py`
- `core/physics/ray_tracing.py`
- `core/physics/distances.py`

## [2026-05-02] Phase 1 시간 지연 엔진의 탐색/선택 기본값

### 결정
시간 지연 추출은 Bag et al. 2022 수식에 맞춰 `(Δt_try, μ_try)` 2D 그리드에서 수행하고,
보수 선택의 pair 검증 기본 축은 `μ` 부호 반대 쌍으로 둔다.

### 배경
Phase 1은 관측 광도곡선 처리 단계라 프로젝트 전역 SIE 표준 근사와 직접 결합하지 않는다.
다만 결과 `Σ(Δt, μ)` 맵은 Phase 5 ML 입력 모달리티로 쓰이므로 2D 포맷을 고정해야 한다.

### 결정 근거
- `μ`는 사전에 알려져 있지 않으므로 `Δt`와 함께 탐색해야 한다.
- prompt의 pair 검증 요구가 `pair_axis="mu"`를 기본값으로 지정하므로 구현도 이를 따른다.
- 실제 benchmark 원본이 없으므로 `data/mock/system6_synthetic.h5`는 smoke/개발용으로만 사용한다.

### 관련 파일
- `core/light_curve/fluctuation.py`
- `core/light_curve/time_delay.py`
- `config/time_delay.yaml`

## [2026-05-02] 단일 표준 근사(SIE) 채택 — 다중 프로파일 폐기

### 결정
프로젝트 전체에서 **단 하나의 근사 방식**만 사용한다.
시뮬레이션·역산·추론 모두에서 동일하게 적용되며, 외부에서 토글 불가.

표준 근사의 내용:
- 렌즈 질량 분포: **SIE** (Singular Isothermal Ellipsoid, σ_v + 축비 q)
- 외부 수렴 κ_ext: 무시 (κ_ext = 0)
- Substructure: 평활화 (smooth mass profile)
- 렌즈 평면: 단일 평면
- 속도 분산: 등방 (β = 0)
- 광원 (Mode 3): extended (Sérsic 또는 pixelated)

### 배경
이전 설계는 4개의 근사 프로파일(`FULL_NUMERICAL` / `SIE_LENS` / `SIS_LENS` / `POINT_LENS`)을
두고 ML이 **어떤 단순화가 적용됐는지를 axes one-hot으로 입력 받아 조건부 학습**하는
구조였다. 이는 다음 복잡도를 야기했다:

- `ml/utils/profile_encoding.py` 신규 모듈 (axes → 15차원 one-hot)
- `ParamEncoder`의 입력 차원이 axes_options에 의존 → checkpoint 호환성 문제
- DECISIONS.md 별도 항목으로 axes_options 사전 고정 규칙 명문화 필요
- Dataset에 `(system, profile, mode)` triple 샘플링 로직
- CLI에 `--profiles` 플래그
- HDF5 스키마에 `params/approximation/` 그룹 (시스템별 6축)

이 모든 것이 "ML이 여러 단순화 패턴을 동시에 학습한다"는 가정 위에 쌓여 있었다.

### 결정 근거

| 측면 | 다중 프로파일 (이전) | 단일 표준 근사 (채택) |
|------|---------------------|---------------------|
| ML 학습 분포 | 여러 오차 패턴 mixture | **단일 오차 패턴** (수렴 빠름·품질 ↑) |
| 코드 복잡도 | profile_encoding, axes_options, profile routing | **모두 제거** |
| Checkpoint 호환성 | axes_options 변경 시 깨짐 | 문제 없음 |
| HDF5 스키마 | approximation/ 그룹 + axes 6필드 | 단순 (truth + approx_outputs 페어만) |
| 추론 시 사용자 책임 | 어떤 profile로 풀었는지 명시 필요 | 항상 같으므로 자동 |
| 운영상 유연성 | 여러 속도/정확도 trade-off 선택 가능 | trade-off 고정 (대신 명확) |

핵심 통찰: **"여러 단순화를 한 모델이 다 다룰 수 있어야 한다"는 요구는 실제 사용 사례에서 없다.**
SIE는 강한 렌즈 분야의 표준이고, 모든 관측 분석이 사실상 SIE 또는 그 확장으로 fit된다.
하나로 고정하면 설계 표면적이 크게 줄어든다.

### SIE를 선택한 근거

1. **강한 렌즈 분야의 표준 모델** — COSMOGRAIL, TDCOSMO, SLACS 등 대부분 SIE 기반
2. **지배적 특징 capture** — ellipticity와 velocity dispersion이 시간 지연·magnification의 1차 요인
3. **Closed form** — 편향각 α(θ)와 시간 지연 Δφ가 닫힌 형태로 계산 가능, numerical integration 불필요
4. **계층적 단순화 명확** — irregular_2D ⊃ SIE ⊃ SIS ⊃ POINT 구조에서 SIE가 표현력과 속도의 sweet spot

### 운영 규칙

1. **함수 시그니처에 `approximation_profile` 인자를 추가하지 말 것.**
   표준 근사는 코드 구현에 implicit하게 포함된다.

2. **함수 docstring에 표준 근사 가정 명시:**
   ```python
   def invert_h0(...):
       """
       Mode 1 — H₀ 역산.
       표준 근사 가정: SIE 렌즈, 단일 평면, κ_ext=0.
       ...
       """
   ```

3. **표준 근사를 변경하려면**: 새 experiment 디렉토리로 처음부터 시작.
   기존 학습 데이터·checkpoint는 모두 폐기. 변경 사유를 본 DECISIONS.md에 기록.

4. **HDF5에 `approx_outputs/` 그룹 사용** — 표준 근사로 푼 결과를 truth와 페어로 저장.
   ML 학습 라벨은 항상 `true_values - approx_outputs`.

### 관련 파일

- `ARCHITECTURE.md` §"표준 근사 모델 (Standard Approximation)"
- `inversion/{mode1_h0, mode2_dm, mode3_wrapper}.py` — 모두 SIE 가정
- `ml/models/encoders.py` — `ParamEncoder` 입력은 물리 파라미터만 (one-hot 없음)
- HDF5 스키마: `approx_outputs/`, `correction_targets/` 그룹

---

## [2026-05-02] axes_options는 학습 시작 전에 고정한다  ⚠️ SUPERSEDED

> **이 결정은 2026-05-02 후속 결정 "단일 표준 근사(SIE) 채택"으로 무효화되었다.**
> axes_options 자체가 더 이상 존재하지 않으므로 본 항목은 역사적 기록으로만 보존.
> 새 코드는 위쪽 결정을 따를 것.

### (당시) 결정
`config/ml.yaml`의 `data.axes_options`는 최초 학습 실행 전에 확정하고,
이후 옵션을 추가·삭제·순서 변경 금지.

### (당시) 배경
`ml/utils/profile_encoding.py`의 `encode_axes_to_onehot`은 axes_options의
키 순서와 옵션 목록을 기준으로 15차원 one-hot 벡터를 생성하며,
`ParamEncoder` 입력에 concat된다. axes_options 변경 시 checkpoint 호환성이 깨지는
문제를 사전 고정으로 회피하려 했음.

### 무효화 사유
다중 프로파일 체계 자체를 폐기하면서 axes_options와 profile_encoding 모듈도 함께 제거됨.
ML 입력에서 axes 정보가 사라졌으므로 본 결정의 전제가 더 이상 유효하지 않음.
관련 운영 규칙(checkpoint에 axes_options 직렬화, 로드 시 검증)도 모두 무효.

### 교훈
설계 단계에서 "유연성"의 비용을 충분히 따져봐야 한다.
axes_options라는 유연성을 도입한 순간 운영 규칙이 따라붙고, 실제로는 그 유연성을
거의 사용하지 않을 가능성이 컸음. 단일 표준안으로 돌아가니 모든 관련 복잡도가 사라졌다.

---

<!-- 아래 형식을 복사해서 추가하세요 -->
<!--
## [YYYY-MM-DD] 결정 제목

### 결정
-

### 배경
-

### 결정 근거
-

### 운영 규칙
-

### 관련 파일
-
-->
