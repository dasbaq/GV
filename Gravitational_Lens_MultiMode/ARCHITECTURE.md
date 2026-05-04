# ARCHITECTURE.md
> 구조가 바뀔 때만 업데이트. 매 세션 읽지 않아도 됨.

---

## 프로젝트 목적

미분해 광도곡선에서 시간 지연 추출 →
유효 굴절률 기반 중력렌즈 시뮬레이션 (다양한 H₀, 암흑물질 분포) →
10만 개 가상 데이터 생성 →
세 가지 역산 Mode (H₀ / DM 분포 / Source 복원) 수행 →
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
   Mode 3: 아인슈타인 링/다중상 이미지 → 원본 source 이미지 복원

⑤ ML 오차 보정
   각 Mode 출력 → 보정된 출력 (full numerical 추정치)
```

파이프라인 간 인터페이스: **HDF5 파일만 사용**

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

## 세 가지 역산 Mode

| Mode | 입력 | 출력 | 상태 |
|------|------|------|------|
| **Mode 1** | Δt_obs, 렌즈 모델, (z_L, z_S) | **H₀** [km/s/Mpc] | 구현 예정 |
| **Mode 2** | Δt_obs, 상의 위치 θ_i, magnification μ_i, (z_L, z_S, H₀) | **DM 분포 파라미터** | 구현 예정 |
| **Mode 3** | 다중상/아인슈타인 링 이미지 I_obs(x, y) | **복원된 source 이미지** S(x, y) | **구현 완료** |

모든 Mode는 표준 근사 위에서 동작한다. ML 보정은 Mode별로 별도의
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
| 광원 | (Mode 3에 한해 의미 있음) | **extended** (Sérsic 또는 pixelated) |

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
  주어진 Δt_obs와 SIE 모델로 추정된 Δφ에 대해
    H₀ = (c · Δφ · (1+z_L)) / (Δt_obs · D̃_Δt(H₀=1))
  비선형 최적화 또는 MCMC로 풀이.

Mode 2 (DM 분포 역산):
  관측 (θ_i, μ_i, Δt_ij)를 likelihood로 두고
  SIE 파라미터 vector p = (σ_v, q, ...)에 대해
    p* = argmin_p Σ ‖observable_i - model_i(p)‖²
  prior: H₀, z_L, z_S 고정.

Mode 3 (Source 복원):
  렌즈 방정식  β = θ - α(θ)  의 역방향 매핑.
  S* = argmin_S ‖I_obs - L · S‖² + λ · R(S)
  L: SIE 기반 lensing operator, R: regularization.
  현재 구현은 semi-linear inversion 기반.
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
├── images/                      # [n_systems, H, W]   ← Mode 3용
│   ├── I_obs                    # 관측(렌즈된) 이미지
│   ├── S_true                   # ground-truth source (full_numerical)
│   ├── psf                      # PSF 커널
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
│   └── fermat_potential  [arcsec²]
│
├── approx_outputs/              # [n_systems]   ← 표준 근사로 푼 결과
│   ├── dt_approx               # SIE 가정 하의 Δt
│   ├── H0_approx               # Mode 1 SIE 역산 결과
│   ├── dm_params_approx        # Mode 2 결과
│   └── S_approx [n_systems, H, W]   # Mode 3 결과
│
└── correction_targets/          # ML 훈련 라벨 = true - approx
    ├── mode1_H0_correction      [km/s/Mpc]
    ├── mode2_dm_correction      [vlen vector]
    └── mode3_source_correction  [n_systems, H, W]
```

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
  - Mode 3 correction은 source-plane `S_true - S_approx`만 저장한다.
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
image_size: 128                    # Mode 3용 픽셀 수 (정사각)
pixel_scale: 0.05                  # arcsec/pix
```

샘플링 방법: Latin Hypercube Sampling (LHS).

---

## ML 모델 구조

**공유 인코더 + Mode별 분기 헤드** 구조. 표준 근사가 고정이므로
조건부 입력(축 one-hot)은 사용하지 않는다.

```
입력 모달리티 (공통)
  ① 광도곡선 시계열   → 1D CNN  (가변 길이, 패딩/마스킹)
  ② 물리 파라미터    → MLP
       - min-max 정규화된 (H0_approx, z_L, z_S, sigma_v, q, ...)
       - 표준 근사로 풀어 얻은 결과(`approx_outputs/`)도 피처에 포함
  ③ Σ(Δt_try, μ_try) 곡선 → 2D CNN  (Δt × μ 그리드)
  ④ 관측 이미지 I_obs → 2D CNN  (Mode 3 활성 시에만 사용)

융합: Cross-attention (3-way 또는 4-way, mode_id에 따라 분기)

Mode별 헤드 (target_mode 입력으로 선택)
  ┌──────────────────────────────────────────────┐
  │ Mode 1 head: MLP → (H0_correction, log_σ)    │  scalar
  │ Mode 2 head: MLP → (dm_correction[K], log_σ) │  vector
  │ Mode 3 head: U-Net 디코더 → S_correction[H,W] │  2D residual
  └──────────────────────────────────────────────┘

손실:
  L = Σ_modes [ L_task(mode) + λ_phys · L_physics(mode) + λ_unc · L_calibration(mode) ]

훈련 전략:
  - 단일 모델로 모든 Mode를 multi-task 학습
  - 표준 근사가 고정이라 모든 학습 샘플이 동일한 오차 분포에서 추출됨 → 학습 효율 ↑
  - 추론 시 target_mode를 받아 해당 head만 활성화
```

---

## Mode 3 — 기존 구현 통합 노트

Mode 3 (source 복원)는 별도로 구현되어 있다. ML 보정은 다음과 같이 통합:

1. 기존 Mode 3 솔버를 표준 근사(SIE) 기반 lensing operator로 호출 → `S_approx` 생성.
2. 시뮬레이션 데이터에서는 `S_true` (full numerical로 동일 시스템을 다시 풀어 얻음)가
   알려져 있으므로 학습 라벨 = `S_true - S_approx`.
3. ML 모델은 잔차 이미지를 예측, 추론 시 `S_corrected = S_approx + S_residual_pred`.
4. 기존 Mode 3 코드는 **수정 금지**. 호출 wrapper만 `inversion/mode3_wrapper.py`에 둔다.
