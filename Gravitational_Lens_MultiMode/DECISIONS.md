# DECISIONS.md
> 설계 결정과 그 근거를 기록. 번복 시 반드시 이유와 날짜를 함께 기재.
> 세션마다 읽지 않아도 됨. 충돌 가능성이 있을 때만 참조.

---

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
