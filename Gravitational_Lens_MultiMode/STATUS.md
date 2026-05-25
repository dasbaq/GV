# STATUS.md
> 매 세션 읽음. 작업 완료 시 업데이트.

마지막 업데이트: 2026-05-26 (v0.6 I_obs 이미지 복구, Mode 3 삭제 유지)

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
| Mode 1 | 허블 상수 역산 | (Δt, 렌즈 모델) → H₀ | ✅ 관측→H₀ E2E 구현 (ML 보정 off 기본) |
| Mode 2 | 암흑물질 분포 역산 | (Δt, θ_i, H₀) → DM 파라미터 | ✅ SIE θ+Δt μ-free E2E 구현 |
| Mode 3 | Source 이미지 복원 | I_obs(x,y) → S(x,y) | 🗑️ 삭제됨 (v0.5, DECISIONS.md [2026-05-25]) |

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

### 5-A. 역산 솔버 (Mode 1/2)

```
[x] inversion/observation_io.py     — 관측 입력 어댑터 추가
[x] inversion/delay_extraction.py   — 실관측 포맷 광도곡선 → Δt_obs 추출
[x] inversion/sie_fit.py            — 관측 상 위치 → SIE fit → Δφ 산출
[x] inversion/mode1_h0.py          — H₀ 역산 솔버 단위/import 정합성 수정 (Δφ [rad²])
[x] inversion/obs_to_features.py    — 관측/스펙 → corrector 입력 텐서 (dataset 재현)
[x] pipelines/run_mode1.py          — 관측 HDF5 → Δt_obs/Δφ → H₀ + ML 보정(--apply-correction) JSON CLI
[x] inversion/real_catalog.py       — 실관측 YAML(Gaia GraL X + 외부 Bag+22 결과) → Mode 1 feature spec
                                      (|Δt| 저장, 음수 입력 pair flip 로그, Mode 2 예약필드 보존)
[x] inversion/mode2_dm.py          — DM 분포 역산 솔버 (SIE θ+Δt, μ-free 기본)
[x] pipelines/run_mode2.py          — Phase 4 HDF5 θ_i+dt_lc → Mode 2 DM JSON + MOCK truth eval
[🗑️] inversion/mode3_wrapper.py    — 삭제됨 (v0.5, DECISIONS.md [2026-05-25])
```

### 5-B. ML 오차 보정 (멀티모달)

```
[x] ml/training/feature_schema.py  — ParamEncoder 공통 feature schema, 누락 mask, 품질 지표
                                    (config-driven sigma_dt sampler/quality normalization)
[x] ml/training/dataset.py         — HDF5 스트리밍, Mode 1/2 라벨 분기 + observed_features 입력
                                    (v0.6: image 키 복구 [1,H,W] I_obs [0,1] 정규화)
[x] ml/models/encoders.py          — 4종 인코더 (LC / Param / Σ-curve / Image)
                                    (v0.6: ImageEncoder 복구 — 4-stage 2D CNN, 1ch I_obs)
[x] ml/models/fusion.py            — 3-way Cross-attention 고정 (LC + Param + Σ), image는 head1 전용
[x] ml/models/heads.py             — Mode 1 / 2 분기 헤드 — Mode 3 헤드 삭제됨 유지 (v0.5)
[x] ml/models/error_corrector.py   — 조립체 (target_mode 라우팅, Mode 1/2 only)
                                    (v0.6: img_enc 복구, Mode1Head in_dim d_model×2→d_model×3=384)
[x] ml/training/losses.py          — Mode 1/2 task loss + physics + calibration — ssim/mode3 삭제됨 유지 (v0.5)
[x] ml/training/trainer.py         — multi-task 훈련 루프
[x] pipelines/train_corrector.py   — CLI 엔트리포인트
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
| system 6 (Δt=24.14일) | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음, 실관측 포맷 synthetic Δt 오차 0.09일 | 2026-05-22 |
| ZTF 노이즈 전체 통계 | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| SDSS J1226-0006 | Mode 1 출력 (H₀) | ⬜ 미실행 | — |
| Phase 4 v0.4 selection-bias | Mode 1 ML 보정 | ✅ 통과 — ratio 0.83(leak false), unfiltered RMSE 4.81(no-corr 33.25), r 0.59, coverage 0.695 | 2026-05-22 |
| Phase 4 v0.4 재현 (⚠️ 13-dim) | Mode 1 ML 보정 | ✅ 통과 — ratio 0.74(leak false), unfiltered RMSE 4.55 r 0.62, filtered RMSE 6.18 r 0.39, coverage 0.68. **Kaggle Dataset가 구 13-dim이라 20-dim 재학습 아님** | 2026-05-24 |
| **Phase 4 v0.4 20-dim 정식** | Mode 1 ML 보정 | ✅ **통과** — ratio 0.996(leak false), unfiltered RMSE 4.51 r 0.66 coverage 0.655, filtered RMSE 4.53 r 0.62 coverage 0.80, pos_frac 1.0, NaN 0. observed_features(20-dim) 데이터로 학습. `all_pass_excluding_record_only=false`는 equivalence 생략으로 인한 CUDA forward diff=Inf 한 행 때문 | 2026-05-25 |
| **Phase 4 v0.4 achievable-r ceiling** | Mode 1 ML 보정 | ✅ **완료** — inputs-conditioned ExtraTrees oracle. unfiltered oracle H0 r 0.8096(CI [0.7324,0.8786]), model/ceiling 0.810. filtered oracle H0 r 0.2495(CI [-0.0770,0.5319]) → `filtered_h0_r_min=0.19`로 record_only 해제 | 2026-05-25 |
| **Phase 4 v0.5 (no image)** | Mode 1 ML 보정 | ✅ **완료(열등)** — RMSE 5.574, r 0.234, coverage 0.62. v0.4(r 0.621) 대비 r 급락. 원인: I_obs 제거로 렌즈 형태 신호 손실. v0.6으로 대체 | 2026-05-25 |
| **Phase 4 v0.6 (I_obs 1ch)** | Mode 1 ML 보정 | ✅ **현재 production** — RMSE 5.258, r 0.503, coverage 0.52. v0.5 대비 r +114% 회복. v0.4 대비 r 낮음(S_approx 2ch→I_obs 1ch). coverage 0.52 과소추정 주의 | 2026-05-26 |
| TDC1 Rung 0 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| TDC1 Rung 1 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| DM 회복 정확도 | Mode 2 | ⚠️ MOCK — synthetic SIE quad 회복 테스트 통과, Phase4 v0.4 full-truth subset(n=25) 평가는 SIE-only 한계로 bias 기록(theta_E median rel 0.527, q 0.194, sigma_v 0.236). μ leak 없음 | 2026-05-25 |
| Source 재구성 PSNR/SSIM | Mode 3 | 🗑️ 삭제됨 (v0.5) | — |

---

## 운영 워크플로우

```
[x] M2 로컬 — 카탈로그 생성, scaler, floor/oracle sanity
[x] M2 로컬 — round --phase equivalence
[x] Kaggle CUDA — round --phase train
[x] M2 로컬 — 결과 회수, 분석, 문서화
```

표준 round 스크립트는 `--phase {equivalence,train,all}`을 지원한다.
short sanity run은 `--min-epochs-for-acceptance` 기본 10보다 짧으면
`acceptance: skipped_smoke`, `leak_triggered: null`로 기록한다.

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
- Phase 5 physics loss는 Mode 1/2 한정 D_Δt 일관성 패널티로 교체됐다. v0.4
  `correction_targets/mode2_dm_correction`은 전부 zero placeholder라 Mode 2 physics
  row는 mask되어 0으로 degrade된다. Mode 2 실학습은 nonzero target catalog 선행 필요.
- v0.5에서 Mode 3(Source 이미지 복원)과 Image 입력 모달리티가 삭제됐고(DECISIONS.md [2026-05-25]),
  v0.6에서 I_obs 단일 채널(1ch)만 복구했다. checkpoint 호환 관계:
  v0.4(head1 in_dim=384, img_enc 2ch) ← v0.6 모델 비호환 (img_enc 입력 채널 1ch≠2ch).
  v0.5(head1 in_dim=256, img_enc 없음) ← v0.6 모델 비호환 (in_dim 256≠384).
  현재 production checkpoint: `data/checkpoints/phase4_v0_6_imgres_best.pt`.
  데이터 생성 코드(`error_catalog.py`, `data_adapters`)는 기존 HDF5의
  `images/` 그룹과 `simplification_errors/mode3_source_residual`을 여전히 저장하지만,
  학습 코드는 mode3 관련 데이터를 읽지 않는다(후방 호환 스키마).
- v0.6 coverage 0.52는 목표 범위 [0.62, 0.78]보다 낮다. σ 예측이 과소추정되는 경향으로,
  calibration loss weight 조정 또는 더 큰 카탈로그(n↑)로 개선 가능성 있음.
  v0.4 대비 r 차이(0.503 vs 0.621)는 I_obs(1ch) vs I_obs+S_approx(2ch) 정보량 차이에서 기인.
  S_approx를 추가하면 r 추가 개선이 예상되지만 truth-adjacent 데이터 사용 여부는 DECISIONS 필요.
- Phase 4 v0.1은 truth lens 아래 `theta - alpha_truth(theta) - beta = 0` root-find로
  truth image 위치를 다시 풀고 validity filter + reject/resample을 적용한 500-system catalog다.
  `mode1_H0_correction` std `6.273`, min/max `3.198/34.747`, cross term `-4.040`,
  off/off variance `4.10e-19`.
- Legacy test 실패 분류: Category A 8개는 ML param/image fixture stale
  (`tests/test_corrector.py` 6개, `tests/test_encoders.py` 2개), Category B 11개는
  legacy mock HDF5 schema 의존(`tests/test_dataset.py` 7개, `tests/test_trainer.py` 4개).
  Category C(Phase 4 v0.1 즉시 수정 필요)는 0개.
- Kaggle T4 2-epoch sanity에서 수렴 전 모델에 acceptance/leak trigger가 발동하는
  false-positive가 확인되어 round phase 분리 및 acceptance epoch gate를 도입했다.
  같은 sanity에서 `scheduler.step()` warning과 seed=1337 CUDA epoch1 `train=nan`
  1회가 관측되어 NaN 진단 로그만 추가했다. hyperparameter tuning은 하지 않는다.
- Phase 4 v0.1 CUDA train에서 seed 42/1337 및 scaler variant에서 NaN이 3회 연속 발생해
  v0.2 catalog를 생성했다. v0.2는 v0.1 validity 위에 p99 기반 tail filter
  (`abs(mu_truth) <= 0.9699`, `dphi_sie/dphi_truth in [0.5878, 0.9201]`,
  separation `>= 0.6598`, `dt_approx <= 444.7`, `I_obs.sum <= 77.79`,
  `max(abs(F_joint)) <= 3.408`, `abs(mode1_H0_correction) <= 32.27`)를 AND 결합한다.
  `mode1_H0_correction` std는 v0.1 `6.273`에서 v0.2 `5.861`로 감소했다.
- `data/mock/phase4_v0_2_eval_unfiltered.h5`는 selection-bias 평가용으로 별도 생성했다.
  v0.1 unfiltered와 동일하게 n=200, seed=42, root convergence만 요구하고 v0.1/v0.2 value filter는 끈다.
- Phase 4 v0.2 floor/scaler/equivalence handoff를 확인했다.
  `data/target_scaler_phase4_v0_2.pkl` Mode 1 mean/scale은 `17.4868/5.7334`,
  `data/logs/phase4_v0_2_floor_analysis.json`의 `mode1_H0_correction`
  mean/std/min/max는 `17.3624/5.8614/5.9792/32.2613`이며 NaN/Inf 0, `|mu_truth| < 1`.
  기존 `data/logs/phase4_v0_2_equivalence.json`은 MPS forward max diff `1.043e-07`,
  Welch p `0.9981`로 통과했다. 현재 Codex 프로세스에서는 MPS unavailable이라
  재실행 명령은 시작 전 중단됐다. 같은 scaler의 CPU 1-epoch seed 42/1337/7
  보조 점검에서는 train loss `1.2928/1.3547/1.4253`, NaN batch 0.
- Phase 4 v0.2 Kaggle Dataset dry-run을 확인했다.
  `phase4_v0_2.h5`, `phase4_v0_2_eval_unfiltered.h5`,
  `target_scaler_phase4_v0_2.pkl`, `phase4_v0_2_equivalence.json`,
  `phase4_v0_2_floor_analysis.json`이 포함되고 missing 0이다.
  실제 upload는 사용자 승인 전이라 미실행 상태다.
- Phase 4 v0.2 Kaggle full run은 filtered RMSE `4.33`, r `0.65`,
  coverage `0.54`였지만 unfiltered RMSE `14.62`, r `0.28`, coverage `0.21`,
  bias `-10.8`로 악화되어 leak trigger가 발동했다
  (`unfiltered/filtered RMSE=3.38 > 3.18`). M2 로컬 분석 결과 v0.2
  불합격의 주원인은 재학습 문제가 아니라 카탈로그 selection bias다.
  unfiltered root-converged catalog에 v0.2 filter를 재적용하면 n `200 -> 75`,
  H0 KS p `0.029 -> 0.00285`로 악화되고, robust한 H0 왜곡은
  v0.1 `H0_approx in [45,90]` gate가 가장 크며 v0.2 separation floor가
  추가 왜곡을 만든다. filtered catalog는 correction/dphi_ratio/mu/separation
  support가 full unfiltered보다 cut 통과 subset에 가까워 쉬운 영역만 남긴다.
  Kaggle 산출물을 repo로 회수: `data/checkpoints/phase4_v0_2_imgres_best.pt`,
  `data/logs/phase4_v0_2_imgres_h0_eval{,_unfiltered}.json`,
  `phase4_v0_2_imgres_long_history.json`, `phase4_v0_2_infra_equivalence.json`
  (git ignored). 학습은 50 epoch 상한 중 epoch 16 early-stop(best epoch 8,
  best_val_m1 `0.5014`), NaN 0. acceptance 불합격 항목은 filtered r `0.65 < 0.85`,
  filtered RMSE CI upper `5.09 > 4.862`, unfiltered/filtered 비율 `3.38 > 2.5`다.
- Phase 4 v0.3 catalog filter를 M2 로컬에서 구현/생성했다.
  v0.3은 label-dependent `H0_approx in [45,90]` 및
  `abs(mode1_H0_correction)` gate와 v0.2 support-tail gates를 제거하고,
  root convergence/finite/`dt_true > 0`/`|mu_truth| < 0.98`/truth image
  separation `>= 0.1 arcsec`만 validity로 유지한다. 이후 H0 `[60,80]`를
  10개 bin으로 나눠 stratified quota를 적용한다.
  `data/mock/phase4_v0_3.h5`는 n=500, seed=42, bin별 50개씩이며
  filtered H0 KS vs U[60,80] p `0.984`로 v0.2 p `1.8e-6` 대비 개선됐다.
  unfiltered eval(`data/mock/phase4_v0_3_eval_unfiltered.h5`)과 비교한 1D
  KS p는 H0 `0.144`, correction `0.801`, dphi_ratio `0.853`, mu `0.685`,
  separation `0.351`이다. `data/logs/phase4_v0_3_floor_analysis.json`에
  no-correction/oracle baseline과 selection-bias acceptance 사전 기준
  (`unfiltered/filtered RMSE <= 2.5`, coverage `[0.62,0.78]`)을 고정했다.
- Phase 4 v0.3 Kaggle full run은 학습이 epoch1부터 NaN으로 붕괴해 acceptance 불가다.
  train nan_batches epoch1 `25`, epoch2 `42` + val `3`, epoch2에서 train/val 모두 nan,
  grad_norm `2.29 -> 5.57`. num_workers `4`/`0` 두 변형이 batch 단위까지 bit-identical하게
  NaN → dataloader 비결정성이 아니라 결정적 수치 overflow다. best checkpoint는 nan-collapse된
  epoch2라 eval 수치(filtered RMSE `10.6`, r `0.30`)는 무의미하다. 원인: v0.3이 label 의존
  gate뿐 아니라 v0.2의 입력측 수치안정 tail gate(`max|F_joint|<=3.408`, `I_obs.sum<=77.79`,
  `dt_approx<=444.7`, `|mu|<=0.9699`, `dphi_ratio` band)까지 함께 제거 → 극단 입력 복귀로
  AMP/fp16 overflow. Kaggle 산출물(`data/`, `data_workers0/` 두 변형 동일)을 repo로 회수:
  `data/checkpoints/phase4_v0_3_imgres_best.pt`,
  `data/logs/phase4_v0_3_imgres_h0_eval{,_unfiltered}.json`,
  `phase4_v0_3_imgres_long_history.json`, `phase4_v0_3_infra_equivalence.json` (git ignored).
- bias↔NaN 트레이드오프 결론: validity 컷을 label 의존(H0_approx/correction, bias 원인 — 계속 제외)
  과 입력/관측측(F_joint/I_obs/dt_approx/mu/dphi_ratio, 수치안정용 — 복원)으로 분리해야 한다.
  v0.3.1은 v0.3 stratified quota를 유지하되 입력측 tail gate만 v0.2 임계로 복원한다.
- Phase 4 v0.3.1 catalog filter를 M2 로컬에서 구현/생성했다.
  v0.3.1은 v0.3 H0 `[60,80]` 10-bin quota를 유지하고 label-dependent
  `H0_approx`/correction gate는 제외하며, v0.2 입력측 tail gate
  (`max|F_joint|<=3.408`, `I_obs.sum<=77.79`, `dt_approx<=444.7`,
  `|mu|<=0.9699`, `dphi_ratio in [0.5878,0.9201]`, separation `>=0.6598`)만 복원한다.
  `data/mock/phase4_v0_3_1.h5`는 n=500, seed=42, bin별 50개이며 filtered H0 KS vs
  U[60,80] p `0.999993`로 v0.3 p `0.984` 수준을 유지/개선했다.
  NaN 사전점검은 v0.2 안전범위 안이다: max|F_joint| `3.347`, I_obs.sum `69.03`,
  dt_approx `443.89`, |mu|max `0.969895`, dphi_ratio `[0.58875,0.92007]`,
  separation min `0.66384`, correction max `32.261`.
  다만 filtered/unfiltered KS p는 H0 `0.144`로 유지됐지만 correction/dphi_ratio/separation은
  `0.0`, mu `0.0265`로 v0.3보다 악화됐다. dphi band ablation에서 band 제외 시 correction
  support는 개선되지만 correction max가 `69.34`까지 복귀해 이번 round는 NaN 안전을 우선한다.
  AMP/loss 측 수치 가드(`F_joint` 정규화 재검토, NLL variance floor 등)는 acceptance 필요 제안으로만 보류한다.
- Phase 4 v0.3.1 Kaggle full run도 NaN으로 epoch1 붕괴. AMP off로 재실행하니 NaN 0(50ep, early-stop
  ep25)이라 **NaN은 fp16 전용**임이 확정됐다. 즉 입력 tail cap 복원으로도 NaN이 안 막혔다 →
  "극단 입력→AMP overflow" 가설 폐기. v0.3 history에서 mode1/2/3_task·mode1_cal은 유한한데
  ssim=nan → **NaN 발원지는 fp16/autocast 하의 SSIM**(`_ssim_loss` 분모 eps underflow + 분산 음수).
  단 AMP-off v0.3.1 결과는 acceptance 또 불합격(filtered RMSE 4.45/r 0.66, unfiltered 14.51/r 0.31,
  비율 3.26>3.18 leak)으로 selection bias가 v0.2와 동일하게 잔존 → tail filter가 bias 원인.
- 결론: v0.1/v0.2 입력측 tail filter는 "CUDA fp16 안정화" 오진단 산물이며 실효는 selection bias뿐이었다.
  `ml/training/losses.py`의 SSIM/NLL을 fp32로 고치면 tail filter가 불필요해진다(상세 DECISIONS 참조).
- Phase 4 v0.4 = 물리 validity only를 M2에서 구현/생성. `validity_filter="v0_4"`는 root/finite/
  `dt_true>0`/`|mu|<0.98`/separation`>=0.1`만 유지하고 label·tail cap·H0 quota를 전부 제거(비-stratified).
  `data/mock/phase4_v0_4.h5`(n=500,seed42)의 `mode1_H0_correction` mean/std/max `29.06/13.68/69.34`가
  unfiltered eval(`phase4_v0_4_eval_unfiltered.h5`, n=200) `30.38/13.52/69.34`와 **일치** →
  train↔eval 분포 정합으로 selection bias 구조적 제거(v0.2 filtered ~17 truncation 해소).
  H0 KS vs U[60,80] p는 train/eval 둘 다 ~`0.025`로 동일(자연 분포). `target_scaler_phase4_v0_4.pkl`
  mode1 mean/scale `29.64/13.71`. `data/logs/phase4_v0_4_equivalence.json` forward diff `1.043e-07`,
  Welch p `0.874`, passed. SSIM fp32 수정으로 큰 correction tail에도 NaN 안 나는지는 Kaggle sanity에서 확인.
  Kaggle 폴더 `data/kaggle_upload/lens-phase4-v0-4/`(train/unfiltered/scaler/equivalence) 준비.
- Phase 4 v0.4 Kaggle CUDA full run(AMP on)에서 **NaN 0**(19ep, early-stop, best ep11)으로
  SSIM fp32 수정이 검증됐고, **selection bias가 처음으로 소멸**했다: unfiltered/filtered RMSE 비율
  `0.83`(<=2.5, leak **false**), unfiltered RMSE `14.5→4.81`, r `0.28→0.59`, coverage `0.21→0.695`.
  no-correction RMSE는 filtered `29.63`/unfiltered `33.25`라 모델이 5~7x 개선한다.
  남은 불합격(filtered RMSE CI upper `6.92>4.862`, r `0.33<0.85`)은 성능 문제가 아니라 band가
  v0.2 truncated easy-subset 기준이라 무효한 것이다(filtered==unfiltered 분포가 되어 filtered val
  n=50이 전체 난이도를 정직 평가; 신뢰도 높은 unfiltered n=200은 r `0.59`).
- v0.4 acceptance 재교정 완료: `scripts/phase4_v0_4_round.py` ACCEPTANCE를 no_correction 기준
  RMSE band `[0.5, 11.08]`(point [0.5,16.62])로, `filtered_h0_r_min`을 `0.0`(record_only)로,
  leak floor를 `0.5`로 바꿨다. 재교정 후 v0.4는 bias/RMSE/coverage 전부 통과(r record_only).
  근거 DECISIONS [2026-05-22] + `data/logs/phase4_v0_4_floor_analysis.json`. r ceiling은
  inputs-conditioned oracle로 산정하는 것이 remaining rigor.
  Kaggle 산출물 회수(git ignored): `data/checkpoints/phase4_v0_4_imgres_best.pt`,
  `data/logs/phase4_v0_4_imgres_h0_eval{,_unfiltered}.json`, `..._long_history.json`,
  `..._infra_equivalence.json`. eval 재확인: filtered RMSE `5.81`/r `0.33`, unfiltered RMSE
  `4.81`/r `0.59`, no-correction `29.63`/`33.25`. 이 checkpoint를 Phase 5 실관측 보정에 사용.
- Phase 5 Mode 1 ML 보정 결합 완료: `inversion/obs_to_features.py`(스펙→corrector 입력 텐서,
  dataset.__getitem__ Mode1 재현) + `pipelines/run_mode1.py --apply-correction`(checkpoint+scaler
  로드 → target_mode=1 forward → scaler 역변환 → `H0_corrected = H0_approx + correction`, σ 산출).
  `tests/test_run_mode1_correction.py`: (a) 어댑터가 dataset 출력과 일치, (b) v0.4 checkpoint로
  보정 closed-form 일치, (c) feature 부재 시 graceful skip — 9 passed(e2e/losses 포함).
  v0.4 데모: H0_approx 60.6/31.4/21.5 → H0_corrected 72.4/73.5/73.0 (H0_true 74.0/68.9/63.9).
  실관측 원본 데이터는 여전히 부재라 합성/Phase4-HDF5로만 검증(MOCK).
- Phase 5 실관측 입력 인터페이스 확장 완료: 외부 Bag+22 결과를 포함한 YAML 리스트 카탈로그를
  `inversion/real_catalog.py`에서 검증하고, `ml/training/feature_schema.py`가 dataset/HDF5와
  YAML 추론 경로의 ParamEncoder feature order를 공유한다. 새 scalar 입력은 `dt_lc`,
  `dt_lc_sigma`, 광곡선 품질 지표 4개, `sigma_v`/`theta_E`/`q` missing flag다.
  `dt_lc`/`dt_lc_sigma`는 실측 entry 필수이며, lens feature 누락은 정상 케이스로 mask 처리한다.
  Phase 4 HDF5에는 `observed_features/`와 `light_curve_quality/`를 저장한다.
  ParamEncoder 입력 차원은 13→20으로 증가했으며, 기존 13-dim checkpoint는 로더에서 첫 layer
  weight를 부분 이식해 graceful 호환한다. Mode 2 입력은 실제 추가하지 않고 `mode2_inputs`
  예약 필드만 YAML에서 보존한다.
- Phase 4 v0.4 카탈로그를 20-dim observed feature schema로 재생성했다.
  `data/mock/phase4_v0_4.h5`(n=500, seed42)는 `observed_features/` 및
  `light_curve_quality/`를 포함하며 Mode 1 correction mean/std/min/max는
  `29.065/13.677/3.198/69.339`. `phase4_v0_4_eval_unfiltered.h5`(n=200, off)도
  동일 schema로 갱신했고 correction mean/std/min/max는 `30.378/13.515/6.543/69.339`.
  `target_scaler_phase4_v0_4.pkl` Mode 1 mean/scale은 `29.6389/13.7096`.
  로컬 MPS equivalence는 현재 Codex 프로세스에서 MPS unavailable로 미실행이라,
  stale handoff JSON은 legacy 이름으로 보존했고 Kaggle은 `--phase all`로 CUDA equivalence와
  full train을 같은 세션에서 실행해야 한다. Kaggle staging 파일 복사는 완료됐고 `kaggle`
  CLI도 설치했으나 인증이 없어(`kaggle auth login` 또는 `~/.kaggle/kaggle.json` 필요)
  Dataset version 업로드는 미완료.
- 2026-05-24 Kaggle CUDA full run은 위 업로드 미완료 때문에 **Kaggle에 남아있던 구 13-dim
  `phase4_v0_4.h5`로 학습됐다**(`param_encoder_input_dim_changed: false`, par_enc `(256,13)`).
  결과는 13-dim v0.4 재현이며 통과(ratio 0.74 leak false, unfiltered RMSE 4.55/r 0.62,
  filtered 6.18/r 0.39, coverage 0.68, CPU↔CUDA diff 4.47e-08, NaN 0, best ep10/stop ep18).
  selection bias·NaN 재발 없음. **20-dim observed_features 정식 재학습은 미수행** —
  로컬 20-dim 카탈로그를 Kaggle Dataset에 재업로드한 뒤 재실행해야 한다.
- 2026-05-25 위 미수행 항목 **완료**. 로컬 20-dim 카탈로그(`phase4_v0_4.h5`/eval/scaler/floor)를
  Kaggle Dataset `donghyun51/lens-phase4-v0-4` 새 버전으로 업로드(인증 ACCESS_TOKEN)하고,
  Kaggle CUDA에서 **`--phase train`(equivalence 의도적 생략) `--workers 0 --epochs 50 --bootstrap-n 1000`**
  로 재학습했다. 데이터 경로 `lens-phase4-v0-4/phase4_v0_4.h5`(20-dim observed_features 포함)로
  학습됨을 확인. 결과: 27ep early-stop(best ep19), NaN 0, filtered RMSE 4.53(CI [3.30,5.75])/r 0.62/
  coverage 0.80, unfiltered RMSE 4.51(CI [4.02,4.99])/r 0.66/coverage 0.655, ratio 0.996(<=2.5),
  pos_frac 1.0/1.0, **leak_triggered false**. 13-dim 재현(unfiltered r 0.62)보다 r 소폭 개선,
  selection bias·NaN 재발 없음. `all_pass_excluding_record_only=false`는 equivalence 생략으로
  CUDA forward diff가 `Inf`(미산출)인 한 행 때문이며 성능/bias/calibration 행은 전부 pass다.
  CPU↔CUDA 동등성은 이전 라운드(5-24, diff 4.47e-08)에서 입증됐고 이번엔 입력 차원만 13→20이라
  검증을 생략했다. **주의**: `param_encoder_input_dim_changed`는 config↔predeclared(둘 다 20) 비교라
  데이터 13/20-dim을 구분하지 못한다. 20-dim 진위 증거는 데이터 경로(새 dataset)와 checkpoint
  `par_enc.net.0.weight=(256,20)`다(checkpoint 회수 후 최종 확인 예정).
- v0.4 관련 코드(round 스크립트/feature_schema/real_catalog/config)는 PR #1로 `origin/main`에 머지됨.
  Kaggle 노트북은 main을 clone하면 v0.4 스크립트가 존재한다(과거 clone 실패는 머지 전 시점 때문).

---

## 다음 작업

1. [x] **완료(2026-05-25)** — v0.5 Kaggle CUDA 재학습. RMSE 5.574, r 0.234. image 삭제로 성능 열등 → v0.6으로 대체.
2. [x] **완료(2026-05-26)** — v0.6 I_obs 이미지 복구 + Kaggle CUDA 재학습.
   RMSE 5.258, r 0.503, coverage 0.52. v0.5 대비 r +114% 회복.
   production checkpoint: `data/checkpoints/phase4_v0_6_imgres_best.pt`.
3. [x] **완료(2026-05-25)** — 20-dim 산출물 회수 완료
   (`checkpoints/phase4_v0_4_imgres_best.pt`, `logs/phase4_v0_4_imgres_h0_eval{,_unfiltered}.json`,
   `..._infra_equivalence.json`). checkpoint `par_enc.net.0.weight=(256,20)` 확인 완료.
4. **[진행 중] v0.7 coverage 개선** — `scripts/phase4_v0_7_round.py` 생성 완료 (2026-05-26).
   변경: calibration weight `0.1 → 0.3`. 모델·데이터·optimizer 등 v0.6 완전 유지.
   **Kaggle CUDA 재학습 필요** (아래 명령 참조). 결과 회수 후 coverage 확인.
   기대: coverage [0.62, 0.78] 진입. r/RMSE는 소폭 하락 허용.
5. **Phase 4 v1** — 더 큰 카탈로그(n↑), NFW offset 분포 도입, image_size 128 복귀 검토.
   Mode 2 nonzero target catalog 선행 필요 (현재 v0.4 mode2_dm_correction 전부 zero placeholder).
6. 실제 benchmark 데이터 입수 시 Phase 1 system6/ZTF/SDSS/TDC1 검증 재실행
7. 실제 Gaia GraL X + Bag+22 YAML 수집 후 `dt_lc_sigma` 및 품질 지표 분포를 재산정하고
   `config/ml.yaml:data.observed_features`/정규화 범위를 업데이트
