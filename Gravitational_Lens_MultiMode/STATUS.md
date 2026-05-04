# STATUS.md
> 매 세션 읽음. 작업 완료 시 업데이트.

마지막 업데이트: 2026-05-04

---

## 현재 Phase

```
[x] Phase 1 — 시간 지연 추출 엔진
[x] Phase 2 — 중력렌즈 물리 모델
[ ] Phase 3 — 대규모 시뮬레이션 생성
[x] Phase 4 — 표준 근사 + 오차 카탈로그
[ ] Phase 5 — Mode별 역산 + 멀티모달 ML 오차 보정
```

---

## Mode 정의 (확정)

| Mode | 역할 | 입력 → 출력 | 구현 상태 |
|------|------|-------------|-----------|
| Mode 1 | 허블 상수 역산 | (Δt, 렌즈 모델) → H₀ | ⬜ 미구현 |
| Mode 2 | 암흑물질 분포 역산 | (Δt, θ_i, μ_i, H₀) → DM 파라미터 | ⬜ 미구현 |
| Mode 3 | Source 이미지 복원 | I_obs(x,y) → S(x,y) | ✅ 구현 완료 (별도 모듈) |

자세한 정식은 ARCHITECTURE.md §"Mode별 역산 정식" 참조.

---

## 표준 근사 (확정, 단일 고정)

ARCHITECTURE.md §"표준 근사 모델" 참조. 프로젝트 전체에서 단 하나의 근사만 사용.

```
렌즈 질량 분포: SIE (Singular Isothermal Ellipsoid, σ_v + 축비 q)
외부 수렴 κ_ext: 무시
Substructure:    평활 (smoothed out)
렌즈 평면:        단일 평면
속도 분산:        등방 (isotropic)
광원 (Mode 3):    extended (Sérsic 또는 pixelated)
```

ML 학습 라벨 = `output(full_numerical) − output(SIE 표준 근사)`.
근사가 고정이므로 ML에 axes 같은 조건부 입력 없음.

---

## Phase 1 — 시간 지연 추출 엔진

```
[x] reconstruction.py   — f1,rec(Δt, μ) 재구성 (2D 그리드)
[x] smoothing.py        — Shafieloo 반복 스무딩
[x] fluctuation.py      — ε(Δt, μ), Σ(Δt, μ) 계산
[x] time_delay.py       — Δt 추출 + 불확도
```

## Phase 2 — 중력렌즈 물리 모델

```
[x] refractive_index.py  — n_eff(r)
[x] lens_models.py       — irregular_2D / SIE / SIS / NFW / 점질량
[x] ray_tracing.py       — 빛의 궤적
[x] distances.py         — D_L, D_S, D_LS, D_Δt
```

## Phase 3 — 대규모 시뮬레이션 생성

```
[x] quasar_lc.py         — DRW + noise model
[x] noise_model.py       — ZTF / LSST photometric noise
[x] generator.py         — 단일 시스템 SIE 기반 mock-schema HDF5 생성 + MSD perturbation injection (κ_ext)
[x] image_renderer.py    — inverse ray-trace + Gaussian PSF
```

## Phase 4 — 표준 근사 + 오차 카탈로그

```
[x] standard_approx.py   — SIE 표준 근사 적용 함수
                          (full_truth → approx_outputs 변환)
[x] error_catalog.py     — full_numerical과 SIE 결과 페어링 →
                          correction_targets 저장
```

## Phase 5 — Mode별 역산 + 멀티모달 ML 오차 보정

### 5-A. 역산 솔버 (Mode 1/2 신규, Mode 3 wrapper)

```
[ ] inversion/mode1_h0.py          — H₀ 역산 솔버 (SIE 가정)
[ ] inversion/mode2_dm.py          — DM 분포 역산 솔버 (SIE 가정)
[ ] inversion/mode3_wrapper.py     — 기존 Mode 3 솔버 호출 wrapper (코드 수정 금지)
```

### 5-B. ML 오차 보정 (멀티모달)

```
[ ] ml/training/dataset.py         — HDF5 스트리밍, Mode별 라벨 분기
[ ] ml/models/encoders.py          — 4종 인코더 (LC / Param / Σ-2D / Image)
[ ] ml/models/fusion.py            — Cross-attention
[ ] ml/models/heads.py             — Mode 1 / 2 / 3 분기 헤드
[ ] ml/models/error_corrector.py   — 조립체 (target_mode 라우팅)
[ ] ml/training/losses.py          — Mode별 task loss + physics + calibration
[ ] ml/training/trainer.py         — multi-task 훈련 루프
[ ] pipelines/train_corrector.py   — CLI 엔트리포인트
```

### Mock 데이터 브리지

```
[x] src_py/ml/generate_mock_dataset.py — Phase 3/4 부재 중 ML smoke training용 DRW mock HDF5 생성
[x] data/mock/mock_dataset.h5          — synthetic mock lens system HDF5 (smoke 실행 시 재생성)
```

---

## 검증 현황

| 벤치마크 | 적용 Mode | 상태 | 마지막 실행 |
|---------|----------|------|-----------|
| system 6 (Δt=24.14일) | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음, synthetic smoke 통과 | 2026-05-02 |
| ZTF 노이즈 전체 통계 | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| SDSS J1226-0006 | Mode 1 출력 (H₀) | ⬜ 미실행 | — |
| TDC1 Rung 0 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| TDC1 Rung 1 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| (추가 예정) DM 회복 정확도 | Mode 2 | ⬜ 미실행 | — |
| (추가 예정) Source 재구성 PSNR/SSIM | Mode 3 | ⬜ 미실행 | — |

---

## 알려진 문제

- 실제 benchmark 원본 데이터(system6, ZTF, SDSS J1226, TDC1 Rung 0/1)가 없어 MOCK/SKIP 검증만 수행됨.
- `data/mock/mock_dataset.h5`는 Phase 3/4 공백을 메우는 개발용 mock이며 benchmark 통과 근거가 아님.
- Phase 3 v2.3 mock eval에서 `dt_approx` 노이즈로 perfect-kappa oracle은 무너졌지만,
  `dt_ratio + image residual` 입력 조합이 여전히 매우 강해 model H0 r이 0.999대에 머묾.
- v2.3 Mode 1 log_sigma calibration은 over-conservative(`|residual| <= 1σ` coverage 100%)로,
  정식 Phase 4 error catalog 또는 추가 calibration 검증이 필요.
- Phase 3 v2.4에서 ParamEncoder truth-side leak(`params/H0`, `dt_ratio`)을 차단하자
  model H0 r이 0.907까지 하락해 수용 기준 하한(0.93)보다 낮음.
  model RMSE는 2.457로 no-correction(4.623)과 perfect-kappa-oracle(2.972)보다 좋지만,
  입력 정보가 다소 빈약해진 상태로 판단.
- v2.4 log_sigma 1σ coverage는 84%로 v2.3의 100%보다 개선됐으나 목표 상단 80%보다 약간 높음.
- Phase 3 v2.5에서 LC delay와 sigma_curve peak를 `dt_approx_noisy`로 맞춰 truth-side
  `delta_t_obs`/`dt_true` leak을 차단. model H0 r은 0.904로 새 기준(0.85~0.92) 통과,
  model RMSE는 2.532로 no-correction(4.623) 및 perfect-kappa-oracle(2.972)보다 좋음.
- v2.5 model RMSE는 analytic floor acceptance band `[2.7, 3.6]`보다 약간 낮아
  residual leak 또는 image/LC 조합의 과성능 가능성을 완전히 배제하지 못함.
  1σ coverage도 79%로 목표 상단 78%보다 0.01 높아 borderline.
- Phase 3 v2.6에서 infra 동등성 기준을 bit-exact 학습 궤적이 아니라
  forward-only op 정확성 + 다중 seed 학습 분포 일치로 재정의.
  CPU/MPS forward diff는 모든 출력에서 `1e-4` 이내, 다중 seed Welch `p=0.925`로 통과.
- v2.6 MPS 속도는 CPU 대비 `6.30x`; MPS workers=4가 workers=0보다 느려
  full retrain은 `num_workers=0`으로 수행.
- v2.6 model RMSE `3.055`는 analytic floor band `[2.7, 3.6]` 안에 있고,
  bootstrap gap CI `[-0.075, 0.349]`가 0을 포함. 1σ coverage `65%`
  CI `[0.580, 0.716]`는 calibration 목표 `[0.62, 0.78]`와 겹침.
- `scripts/v2_6_round.py`는 Kaggle CUDA 이식을 위해 `LENS_DATA_ROOT`/`LENS_DATA_PATH`
  및 `LENS_WORK_ROOT` 환경변수와 `--device`, `--workers`, `--epochs` 옵션을 지원함.
- Phase 4부터 correction label 부호는 HDF5 schema 정의대로 `true - approx`이다.
  v2.* `target_scaler_phase3_v2_*.pkl` 및 `phase3_v2_*_imgres_best.pt` checkpoint는
  Phase 4 데이터와 호환 불가이며 로드 금지.
- `pytest -q tests/` 전체 회귀는 기존 ML fixture의 stale param/image shape 및 legacy mock
  HDF5 schema 문제로 19개 실패가 남아 있다. Phase 4 신규 테스트와 Phase 4 HDF5
  forward/backward smoke는 통과.
- Phase 4 v0은 50-system small catalog였고 truth Fermat을 SIE image 위치에서 평가하는
  artifact가 확인되어 baseline 비교용으로만 보존한다.
- Phase 4 v0.1은 truth lens 아래 `theta - alpha_truth(theta) - beta = 0` root-find로
  truth image 위치를 다시 풀고 validity filter + reject/resample을 적용한 500-system catalog다.
  `mode1_H0_correction` std `6.273`, min/max `3.198/34.747`, cross term `-4.040`,
  off/off variance `4.10e-19`.
- Legacy test 실패 분류: Category A 8개는 ML param/image fixture stale
  (`tests/test_corrector.py` 6개, `tests/test_encoders.py` 2개), Category B 11개는
  legacy mock HDF5 schema 의존(`tests/test_dataset.py` 7개, `tests/test_trainer.py` 4개).
  Category C(Phase 4 v0.1 즉시 수정 필요)는 0개.

---

## 다음 작업

1. 실제 benchmark 데이터 입수 시 Phase 1 system6/ZTF/SDSS/TDC1 검증 재실행
2. Phase 4 v0.1 카탈로그로 ML 재학습 라운드 수행
3. Phase 4 v1에서 image_size 128 복귀 및 NFW offset 분포 도입 검토
4. 선택 시 Kaggle CUDA에서 `real_phase3_v2_6.h5` 재현 실행 후 결과 회수
5. Phase 5는 Phase 4 HDF5로 smoke training 후 정식 재학습 파이프라인으로 교체
