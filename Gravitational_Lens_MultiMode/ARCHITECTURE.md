# ARCHITECTURE.md
> 구조가 바뀔 때만 업데이트. 매 세션 읽지 않아도 됨.

---

## 프로젝트 목적

미분해 광도곡선에서 시간 지연 추출 →
유효 굴절률 기반 중력렌즈 시뮬레이션 (다양한 H₀, 암흑물질 분포) →
10만 개 가상 데이터 생성 →
두 가지 역산 Mode (H₀ / DM 분포) 수행 →
**표준 근사** 적용 + 멀티모달 ML로 근사 오차 보정

핵심 아이디어: 실제 우주의 복잡한 구조(불규칙한 은하 질량 분포, 확장된 광원,
substructure, 다중 평면 효과 등)를 매번 numerically 풀면 너무 느리다.
대신 **하나의 표준 근사**(SIE 타원 + 단일 평면 등)로 빠르게 풀고,
근사로 잃은 정보를 ML이 복원한다.
근사 방식이 프로젝트 전체에서 동일하게 고정되므로 ML은 **하나의 오차 패턴**에 집중하여 학습한다.

---

## 파이프라인 흐름

```
① 시간 지연 추출
   F(t) → f1,rec(Δt, μ) → ε(Δt, μ) → Σ(Δt, μ) → Δt

② 중력렌즈 물리 모델 (정방향)
   n_eff(r) → 빛의 궤적 → μ, Δt_phys, 상의 위치/형태

③ 대규모 시뮬레이션
   H₀ × DM분포 × z_L × z_S → 10만 개 HDF5 (표준 근사 적용)
   (학습용 페어: full numerical 결과를 동시에 저장)

④ Mode별 역산 (Inversion)
   Mode 1: 관측 Δt + 렌즈 모델 → H₀
   Mode 2: 관측 (Δt, 상의 위치, μ) → DM 질량 분포 파라미터
   Mode 3: 🗑️ 삭제됨 (v0.5, DECISIONS.md [2026-05-25])

⑤ ML 오차 보정
   각 Mode 출력 → 보정된 출력 (full numerical 추정치)
```

파이프라인 간 정식 학습/평가 인터페이스: **HDF5 파일 사용**.
실관측 1단계 ingest는 사람이 관리하는 **YAML 리스트**를 허용한다. YAML은
Gaia GraL X 메타데이터와 외부 Bag+22 결과(`dt_lc`, `dt_lc_sigma`, 광곡선 품질 지표)를
담는 entry 포맷이며, 코드 내부에서는 공통 feature schema로 변환되어 HDF5 dataset 경로와
같은 ParamEncoder 입력 순서를 사용한다. Bag+22 raw 추출과 ZTF 다운로드는 이 ingest
adapter의 책임이 아니다.

실관측/시뮬레이션 공통 ParamEncoder 입력 정책은 `config/ml.yaml:data.observed_features`와
`data.param_normalization`이 단일 source of truth다. Δt 저장은 `|Δt|` 양수만 허용하고,
음수 YAML 입력은 configured policy로 pair order를 뒤집어 메타데이터 로그에 남긴다. 시뮬레이션
`sigma_dt` proxy는 config의 `relative_then_clip` sampler에서 생성하며, 광곡선 품질 지표는
config의 `transform`(기본 log)과 min/max 범위로 정규화한다. 누락 가능한 렌즈 scalar는
normalized-zero sentinel과 field별 missing flag로 표현한다.

### 실제 관측 Mode 1 ingestion

raw 관측 파일은 바로 역산기에 넣지 않고, 먼저 observation HDF5로 정규화한다.

```
CSV/TSV/RDB light curves + YAML manifest + optional sidecar
  → pipelines/ingest_observation.py
  → data/observations/*_observed.h5
  → pipelines/run_mode1.py
```

- light curve table은 A/B 두 상의 공통 `t_obs` grid를 가져야 한다.
- magnitude 입력은 `F = 10^(-0.4 m)`로 선형 flux로 변환하고,
  `sigma_F = 0.4 ln(10) F sigma_m`으로 오차를 전파한다.
- manifest에는 `image_positions` [arcsec], `z_lens`, `z_source`, column mapping을 둔다.
- 검증용 `dt_ref_days`, `H0_ref` 등은 sidecar와 ingestion report에만 두고,
  Mode 1 입력 HDF5에는 저장하지 않는다.
- 실제 benchmark 순서는 TDC1 Rung 0의 Δt 추출기 검증 후 SDSS J1226-0006 E2E H0 검증이다.

---

## 워크플로우 — M2 전처리 / Kaggle GPU 학습

정식 라운드는 환경별 역할을 분리한다.

| 단계 | 실행 환경 | 진입점 | 산출물 |
|---|---|---|---|
| 시뮬레이션/카탈로그 생성 | M2 Air 로컬 | `scripts/build_*.py` 또는 `src_py/simulation/generator.py` | `data/mock/*.h5` |
| 라벨 분포 sanity, floor 분석 | M2 Air 로컬 | `scripts/floor_analysis_*.py` 또는 라운드별 분석 코드 | `data/logs/*.json`, `data/target_scaler_*.pkl` |
| ML 학습 + bootstrap | Kaggle CUDA | `scripts/<round>_round.py` | checkpoint, eval JSON |
| 결과 분석/문서화 | M2 Air 로컬 | 직접 분석 | 문서, 로컬 ignored artifacts |

표준 round 스크립트는 다음 환경변수를 지원한다.

| 변수 | 의미 |
|---|---|
| `LENS_DATA_PATH` | 학습/검증 HDF5 절대경로 |
| `LENS_DATA_PATH_UNFILTERED` | selection bias 평가용 HDF5 절대경로 |
| `LENS_DATA_ROOT` | 개별 path 미지정 시 fallback 디렉토리 |
| `LENS_WORK_ROOT` | checkpoint/log/scaler 출력 prefix |
| `LENS_SCALER_PATH` | scaler pkl 절대경로. 명시 시 read-only input으로 취급 |

`data/`의 운영 구조:

```
data/
├── mock/                 # 카탈로그(.h5), Kaggle Dataset 업로드 대상
├── checkpoints/          # 학습 산출물, M2 <-> Kaggle 동기화
├── logs/                 # 평가 JSON, Kaggle 출력 회수 대상
├── runs/                 # tensorboard 등 임시 산출물
└── target_scaler_*.pkl   # scaler, 카탈로그와 함께 업로드
```

`.h5`, `.pkl`, `.pt` 및 위 artifact 디렉토리는 git에 커밋하지 않는다.
공유는 Kaggle Dataset을 통해 수행하고, 결과는 Kaggle output에서 회수한다.
round 스크립트는 `--phase {equivalence,train,all}` dispatcher로 M2 equivalence와
Kaggle train/eval을 분리하며, smoke run은 acceptance/leak 판정을 건너뛴다.

---

## 두 가지 역산 Mode

| Mode | 입력 | 출력 | 상태 |
|------|------|------|------|
| **Mode 1** | Δt_obs, 렌즈 모델, (z_L, z_S) | **H₀** [km/s/Mpc] | ✅ 구현 완료 |
| **Mode 2** | Δt_obs, 상의 위치 θ_i, magnification μ_i, (z_L, z_S, H₀) | **DM 분포 파라미터** | ✅ 구현 완료 |
| ~~Mode 3~~ | ~~I_obs(x, y)~~ | ~~source S(x, y)~~ | 🗑️ **삭제됨** (v0.5, DECISIONS.md [2026-05-25]) |

Mode 1/2는 표준 근사 위에서 동작한다. ML 보정은 Mode별로 별도의
head를 가지며 인코더는 공유한다.

---

## 표준 근사 모델 (Standard Approximation)

프로젝트 전체에서 단 하나의 근사 방식을 사용한다.
시뮬레이션·역산·추론 모두에서 동일하게 적용되며, ML은 이 한 가지 근사로 인한
오차 패턴만 학습한다.

### 근사의 구체적 내용

| 단순화 항목 | 실제 우주 | 표준 근사 적용 후 |
|-------------|-----------|------------------|
| 렌즈 질량 분포 | 불규칙한 2D pixel-grid Σ(x,y) | **SIE** (Singular Isothermal Ellipsoid) — σ_v, 축비 q |
| 외부 수렴 κ_ext | LOS 효과 modeled | **무시** (κ_ext = 0) |
| Substructure | sub-halo, dark satellites 포함 | **평활화** (smooth mass profile) |
| 렌즈 평면 | 다중 평면 | **단일 평면** |
| 속도 분산 | anisotropic | **isotropic** (β = 0) |
| 광원 | (Mode 3 삭제됨 — 해당 없음) | (해당 없음) |

Thin-lens SIE helper는 deflection과 Fermat potential이 같은 elliptical-potential
근사에서 나오도록 고정한다. 이는 full numerical truth가 아니라 Mode 1/2의 빠른
표준 근사이며, q=1에서는 SIS 식으로 환원된다.

### 왜 SIE 인가

1. 강한 렌즈 분야의 **표준 모델** — 대부분의 관측 분석이 SIE 또는 그 확장으로 fit
2. 지배적 특징(ellipticity + velocity dispersion)을 capture
3. **편향각 α(θ)와 시간 지연 Δφ가 닫힌 형태로 계산** — numerical integration 불필요
4. SIS의 자연스러운 확장이라 단순화 단계가 명확

### ML 보정 방식

- 학습 라벨: `output(full_numerical) − output(standard_approx)`
- 학습 입력: 관측 + 표준 근사 결과 (근사 자체는 항상 같으므로 입력 피처로 인코딩 불필요)
- 추론:
  ```
  result_approx = simulate(input)    # 표준 근사로 풀이
  correction = ML(result_approx, observations, mode_id)
  result_corrected = result_approx + correction
  ```

표준 근사가 고정되어 있으므로 ML 모델은 어떤 단순화가 적용됐는지 알 필요가 없다.
**근사 axes one-hot 같은 조건부 입력은 사용하지 않는다.**

### H0-독립 Fermat-ratio 연구 트랙

기존 Mode 1 H0 correction과 별도로, `y_phi = log(dphi_truth/dphi_SIE)`의 조건부
posterior를 학습한다. 이 트랙의 입력은 SIE 구조 파라미터, `theta_E`로 정규화한 두 상 위치,
관측 이미지뿐이다. H0, H0_approx, 절대 시간 지연, 광도곡선은 입력·label·loss에서 제외한다.
`run_mode1 --apply-phi-correction`의 H0 표시는 posterior를 downstream 물리식에 넣은
diagnostic일 뿐, benchmark 판정에는 사용하지 않는다.

### Mode 1 ML inference reliability metadata

Mode 1 ML 보정은 point estimate를 적용하기 전에 inference-side domain-membership을
평가한다. 이 계층은 Phase4 v0.4 catalog profile과 실측 입력에서 직접 만들 수 있는
ParamEncoder feature, 광도곡선 tail, image availability, Δt/μ guard만 사용한다.
`mu_truth`, `dphi_sie/dphi_truth`, correction label 같은 truth-only 값은 사용하지 않는다.

결과 JSON의 `ml_correction.domain_membership`에는 `domain_score`, `domain_grade`
(`in_distribution`, `borderline`, `ood_abstain`), `failed_checks`,
`sigma_scale_regime`, `profile_artifact`, `benchmark_use`가 기록된다. `borderline`은
보정을 적용할 수 있지만 conservative sigma multiplier와 `benchmark_use=false`를
동반하며, `ood_abstain`은 H0 point correction을 적용하지 않는다.

---

## 핵심 물리

### 시간 지연 추출 (Bag et al. 2022, arXiv:2110.15315)

```
F(t) = f1(t) + μ · f1(t - Δt)

f1,rec(t; Δt_try, μ_try) = Σ_{n=0}^∞ (-μ_try)ⁿ · F(t - n·Δt_try)
            수렴 조건: |μ_try| < 1  (더 밝은 이미지 = image 1)

ε(Δt_try, μ_try) = Σᵢ [f1,rec(tᵢ) - f1,rec(tᵢ₊₁)]²

Σ(Δt_try, μ_try) = (ε - <ε>) / σ_ε        # (Δt, μ) 그리드 전체 통계로 정규화

식별 기준:
  보수: Σ < -2.0  (쌍 극솟값 둘 다 — Δt 축에서 짝지어진 두 극솟값 모두 통과)
  완화: Σ < -1.0  + 깊이 baseline의 50% 이상
```

스윕 차원이 (Δt_try, μ_try) 2D인 이유: μ는 사전에 알려져 있지 않으므로 같이 탐색해야 함.
구현 시 (Δt, μ) 2D 그리드 일괄 처리 = 벡터화 (Phase 1 prompt 참조).

### 유효 굴절률 및 시간 지연

```
n_eff(r) = 1 - 2Φ(r)/c²

Phase 2 중심 계산 흐름:
  질량 모델 / 중력 퍼텐셜 Φ
      → n_eff = 1 - 2Φ/c²
      → ∇n_eff 또는 ∇Φ
      → n_eff field 기반 ray tracing
      → optical path length ∫ n_eff ds
      → travel time = OPL / c
      → path 간 time delay

편향각: α = (2/c²) ∫ ∇⊥Φ dl

페르마 포텐셜: φ(θ) = ½|θ - β|² - ψ(θ)

시간 지연: Δt = (1 + z_L) · D_Δt/c · Δφ

D_Δt = (1 + z_L) · (D_L · D_S) / D_LS

H₀ ∝ 1 / D_Δt              ← Mode 1 역산의 핵심 관계
```

### 렌즈 모델

| 모델 | 파라미터 | 포텐셜 Φ | 용도 |
|------|---------|---------|------|
| irregular_2D | pixel-grid Σ(x,y) | 수치 적분 | 시뮬레이션 ground truth (full numerical) |
| **SIE** | **σ_v, 축비 q** | **SIS 확장 (closed form)** | **표준 근사 — 모든 역산이 이 모델로 풀이** |
| SIS | σ_v [km/s] | σ_v² · ln(r) | SIE의 q=1 특수해 |
| NFW | M₂₀₀, c | 수치 적분 | DM halo 모델 (truth 생성용) |
| 점질량 | M [M☉] | -GM/r | 극단 케이스 (디버깅·해석해 검증용) |

### Mode별 역산 정식

```
Mode 1 (H₀ 역산):
  주어진 Δt_obs와 SIE 모델로 추정된 Δφ [rad²]에 대해
    H₀ = (c · Δφ · (1+z_L)) / (Δt_obs · D̃_Δt(H₀=1))
  비선형 최적화 또는 MCMC로 풀이.

Mode 2 (DM 분포 역산):
  관측 (θ_i, μ_i, Δt_ij)를 likelihood로 두고
  SIE 파라미터 vector p = (σ_v, q, ...)에 대해
    p* = argmin_p Σ ‖observable_i - model_i(p)‖²
  prior: H₀, z_L, z_S 고정.

Mode 3 (Source 복원): 🗑️ 삭제됨 (v0.5, DECISIONS.md [2026-05-25])
```

---

## HDF5 데이터 스키마

```
simulation_YYYYMMDD_HHMMSS.h5
├── metadata/
│   ├── created_at, n_systems, git_commit, random_seed
│   └── full_truth_available     # bool — full_numerical 결과가 페어로 저장됐는지
│
├── params/                      # [n_systems]
│   ├── H0          [km/s/Mpc]
│   ├── z_lens, z_source
│   ├── lens_truth_model (string)   # 시뮬레이션의 ground-truth 모델
│   │                                # ("irregular_2D" | "SIE" | "NFW" 등)
│   ├── sigma_v     [km/s]
│   ├── M200        [M☉]
│   ├── concentration
│   └── q                         # 축비 (SIE/SIE-truth 시)
│                                 # ※ 표준 근사의 모델은 항상 SIE 고정이라 별도 필드 불필요
│
├── light_curves/                # [n_systems, max_epochs]
│   ├── F_joint, sigma_noise
│   ├── t_obs       [days]
│   └── n_epochs    (int)
│
├── observed_features/           # [n_systems] inference-side scalar inputs
│   ├── dt_lc                   # 외부/합성 Bag+22 primary-pair Δt [days]
│   ├── dt_lc_sigma             # σ_Δt [days]
│   ├── n_epochs_quality
│   ├── baseline_days
│   ├── median_cadence_days
│   └── median_photometric_error
│
├── light_curve_quality/         # observed_features 품질 지표 alias
│   ├── n_epochs_quality
│   ├── baseline_days
│   ├── median_cadence_days
│   └── median_photometric_error
│
├── images/                      # [n_systems, H, W]   ← 데이터 생성 코드가 여전히 저장
│   ├── I_obs                    # (학습 코드는 읽지 않음 — v0.5에서 Mode 3 삭제됨)
│   ├── S_true
│   ├── psf
│   └── pixel_scale [arcsec/pix]
│
├── true_values/                 # [n_systems]   ← Mode 1 / 2 라벨 (full_numerical 기반)
│   ├── dt_true     [days]
│   ├── mu_true
│   ├── theta_E     [arcsec]
│   ├── H0_true     [km/s/Mpc]    ← Mode 1 라벨
│   ├── dm_params_true            ← Mode 2 라벨 (vlen vector)
│   └── D_delta_t   [Mpc]
│
├── ray_paths/                   # [n_systems]
│   ├── theta_1, theta_2  [arcsec]
│   └── fermat_potential  [rad²]
│
├── approx_outputs/              # [n_systems]   ← 표준 근사로 푼 결과
│   ├── dt_approx               # SIE 가정 하의 Δt
│   ├── H0_approx               # Mode 1 SIE 역산 결과
│   ├── dm_params_approx        # Mode 2 결과
│   └── S_approx [n_systems, H, W]   # Mode 3 결과 — 학습 코드는 읽지 않음 (v0.5)
│
└── correction_targets/          # ML 훈련 라벨 = true - approx
    ├── mode1_H0_correction      [km/s/Mpc]
    ├── mode2_dm_correction      [vlen vector]
    └── mode3_source_correction  [n_systems, H, W]   ← 데이터 생성만; 학습 코드는 읽지 않음 (v0.5)
```

### 실관측 YAML ingest schema

실관측 1단계 카탈로그는 top-level YAML list이며 entry마다 다음 nested dict를 사용한다.

```
- name: <system id>
  sources: {...}
  redshifts: {z_lens: ..., z_source: ...}
  kinematics: {sigma_v: null | <km/s>}       # 누락 가능
  lens_model:
    H0_approx: null | <km/s/Mpc>
    theta_E: null | <arcsec>                 # 누락 가능
    q: null | <axis ratio>                   # 누락 가능
    dphi_rad2: null | <rad^2>
  time_delay:
    dt_lc: <days>                            # 필수
    dt_lc_sigma: <days>                      # 필수
    image_pair_convention: brightest_pair_positive_delay
  light_curve_quality:
    N_epochs: ...
    baseline_days: ...
    median_cadence_days: ...
    median_photometric_error: ...
  mode2_inputs:                              # 1단계에서는 보존만 함
    image_positions: null
    all_pair_delays: null
    flux_ratios: null
    magnifications: null
    is_lens_probability: null
```

`dt_lc`는 현재 시뮬레이션 규약과 맞춰 primary pair의 양수 시간지연으로 둔다.
시뮬레이션 primary pair는 magnification이 큰 두 상(`theta_1`, `theta_2`)이다.
`sigma_v`, `theta_E`, `q`는 실측에서 정상적으로 누락될 수 있으며 ParamEncoder에는
normalized-zero sentinel과 missing flag(`missing_sigma_v`, `missing_theta_E`, `missing_q`)로 들어간다.
`M200`, `concentration`, `kappa_ext`, `nfw_offset` 같은 truth-only 키는 YAML ingest에서 거부한다.

학습 데이터 생성: 동일한 (H₀, z_L, z_S, …) 시스템을 두 번 풀이 —
한 번은 full numerical (truth), 한 번은 SIE 표준 근사. 차이를 `correction_targets/`에 저장.

### Phase 4 구현 모듈

- `core/physics/standard_approx.py`
  - 프로젝트 전역 단일 SIE 표준 근사를 적용한다.
  - 입력은 공개 inference-side 키(`H0`, `z_lens`, `z_source`, `sigma_v`, `q`,
    `source_pos_xy`)만 사용한다.
  - `M200`, `concentration`, `kappa_ext`, `nfw_offset`는 truth-only 키이므로
    표준 근사 함수에 전달되면 명시적으로 거부한다.
  - Mode 1 1차 구현은 SIE Fermat potential 차이와 관측 full-truth delay로
    closed-form H0 inversion을 수행한다.

- `ml/data/error_catalog.py`
  - full truth와 SIE 표준 근사의 페어를 생성하고 HDF5 schema의
    `true_values/`, `approx_outputs/`, `correction_targets/`를 채운다.
  - Phase 4 v0 truth는 deflection-additive:
    `alpha_truth(theta) = alpha_SIE(theta) + alpha_NFW(theta) + kappa_ext * theta`.
  - Phase 4 v0.1부터 full truth 시간 지연은 SIE image 위치가 아니라
    truth lens 아래에서 수치적으로 푼 image position에서 평가한다:
    `theta_truth = root(theta - alpha_truth(theta) - beta)`.
    root 초기값은 SIE-only image이며, v0.1은 SIE-anchored search만 수행한다.
    truth-only extra image의 전역 탐색은 향후 라운드로 남긴다.
  - NFW는 origin-aligned, `kappa_ext`는 `[0, 0.1]`, off-mode는 flag로 제어한다.
  - validity filter는 root 수렴, finite 값, `dt_true > 0`, `abs(mu_truth) < 0.98`,
    truth image separation `>= 0.1 arcsec`, `H0_approx in [45, 90]`,
    `dphi_sie / dphi_truth in [0.5, 1.5]`를 요구한다.
  - Mode 2 correction은 정식 solver 전까지 zeros 유지.
  - Mode 3 correction(`S_true - S_approx`)은 데이터 생성 코드에서만 저장; 학습 코드는 읽지 않음 (v0.5).
  - v2.* ML 호환을 위해 `simplification_errors/` alias도 저장하지만,
    부호는 Phase 4 schema와 동일한 `true - approx`이다.

---

## 시뮬레이션 파라미터 범위

```yaml
H0:        [60, 80]    km/s/Mpc
z_lens:    [0.1, 1.0]
z_source:  [0.5, 3.5]  (z_S > z_L + 0.1)
sigma_v:   [150, 350]  km/s
M200:      [1e12, 1e14] M☉         # NFW / irregular_2D truth
c_nfw:     [3, 15]
q:         [0.6, 1.0]              # SIE 축비
SF_inf:    [0.1, 0.5]  mag         # DRW 진폭
tau_drw:   [100, 1000] days        # DRW 시간 스케일
image_size: 128                    # HDF5 데이터 생성용 (학습 코드는 읽지 않음, v0.5)
pixel_scale: 0.05                  # arcsec/pix
```

샘플링 방법: Latin Hypercube Sampling (LHS).

---

## ML 모델 구조

**공유 인코더 + Mode별 분기 헤드** 구조. 표준 근사가 고정이므로
조건부 입력(축 one-hot)은 사용하지 않는다.

```
입력 모달리티 (공통, v0.5 이후 3종 고정)
  ① 광도곡선 시계열   → 1D CNN  (가변 길이, 패딩/마스킹)
  ② 물리 파라미터    → MLP
       - min-max 정규화된 (H0_approx, z_L, z_S, sigma_v, q, ...)
       - 표준 근사로 풀어 얻은 결과(`approx_outputs/`)도 피처에 포함
  ③ Σ(Δt_try, μ_try) 곡선 → 1D CNN  (피크 주변 단면)
  ④ 관측 이미지 I_obs: 🗑️ 삭제됨 (v0.5, DECISIONS.md [2026-05-25])

융합: 3-way Cross-attention 고정 (LC + Param + Σ-curve)
  fused = CrossAttentionFusion(h_lc, h_params, h_sigma).mean(dim=1)

Mode별 헤드 (target_mode 입력으로 선택)
  ┌──────────────────────────────────────────────────────────────────┐
  │ Mode 1 head: MLP(in_dim=d_model×2) → (H0_correction, log_σ)     │  scalar
  │              in = cat([fused, h_lc])  — d_model×3 아님 (v0.5 변경) │
  │ Mode 2 head: MLP → (dm_correction[K], log_σ)                    │  vector
  │ Mode 3 head: 🗑️ 삭제됨 (v0.5)                                    │
  └──────────────────────────────────────────────────────────────────┘

손실 (v0.5):
  L = Σ_{mode∈{1,2}} [ L_task(mode) + λ_phys · L_physics(mode) + λ_unc · L_calibration(mode) ]
  (mode3_task, ssim 항목 삭제됨)

훈련 전략:
  - 단일 모델로 Mode 1/2를 multi-task 학습
  - 표준 근사가 고정이라 모든 학습 샘플이 동일한 오차 분포에서 추출됨 → 학습 효율 ↑
  - 추론 시 target_mode를 받아 해당 head만 활성화
  - v0.4 checkpoint는 Mode1Head in_dim=384(d_model×3)로 v0.5(256)와 비호환 → 재학습 필요
```

---

## Mode 3 — 삭제 이력

> **v0.5 (2026-05-25)**: Mode 3(Source 복원)과 Image 입력 모달리티가 삭제됨.
> 상세 근거와 삭제 전 ablation 기록은 DECISIONS.md [2026-05-25] 참조.

삭제된 구성 요소:
- `ml/models/encoders.py`: `ImageEncoder`, `_Conv2DBlock` 클래스
- `ml/models/heads.py`: `Mode3Head`, `_UpBlock` 클래스
- `ml/models/fusion.py`: 4-way 분기 경로 (image 포함)
- `ml/training/losses.py`: `_ssim_loss`, `mode3_task` 항목
- `ml/training/dataset.py`: `image`, `use_image`, `target_image` 키
- `inversion/mode3_wrapper.py`: 파일 전체 삭제
- `config/ml.yaml`: `image_size`, `image_backbone`, `mode3`/`ssim` loss weight

데이터 생성 코드(`ml/data/error_catalog.py`)는 기존 HDF5의 `images/` 그룹과
`simplification_errors/mode3_source_residual`을 계속 저장하지만,
학습 코드는 이를 읽지 않는다 (후방 호환 HDF5 스키마 보존).
