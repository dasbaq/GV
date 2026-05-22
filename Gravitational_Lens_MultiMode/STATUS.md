# STATUS.md
> 매 세션 읽음. 작업 완료 시 업데이트.

마지막 업데이트: 2026-05-22

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
[x] inversion/observation_io.py     — 관측 입력 어댑터 추가
[x] inversion/delay_extraction.py   — 실관측 포맷 광도곡선 → Δt_obs 추출
[x] inversion/sie_fit.py            — 관측 상 위치 → SIE fit → Δφ 산출
[x] inversion/mode1_h0.py          — H₀ 역산 솔버 단위/import 정합성 수정 (Δφ [rad²])
[x] pipelines/run_mode1.py          — 관측 HDF5 → Δt_obs/Δφ → H₀ JSON CLI
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
| system 6 (Δt=24.14일) | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음, 실관측 포맷 synthetic Δt 오차 0.09일 | 2026-05-22 |
| ZTF 노이즈 전체 통계 | Phase 1 입력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| SDSS J1226-0006 | Mode 1 출력 (H₀) | ⬜ 미실행 | — |
| TDC1 Rung 0 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| TDC1 Rung 1 | Mode 1 출력 | ⚠️ MOCK/SKIP — 원본 데이터 없음 | 2026-05-02 |
| (추가 예정) DM 회복 정확도 | Mode 2 | ⬜ 미실행 | — |
| (추가 예정) Source 재구성 PSNR/SSIM | Mode 3 | ⬜ 미실행 | — |

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

---

## 다음 작업

1. 실제 benchmark 데이터 입수 시 Phase 1 system6/ZTF/SDSS/TDC1 검증 재실행
2. Kaggle CUDA에서 Phase 4 v0.3.1 full train을 수행한다:
   `python scripts/phase4_v0_3_1_round.py --phase train --equivalence-from /kaggle/input/<slug>/phase4_v0_3_1_equivalence.json --device cuda --workers 4 --epochs 50 --bootstrap-n 1000`.
   acceptance 사전 기준은 selection-bias ratio `<=2.5`, 1σ coverage `[0.62,0.78]`이다.
3. Phase 4 v0.3에서 coverage 개선은 NLL/calibration 문제로 별도 점검한다
   (filtered validation뿐 아니라 unfiltered/cut-boundary bin별 residual/sigma 검증).
4. Phase 4 v1에서 image_size 128 복귀 및 NFW offset 분포 도입 검토
5. 선택 시 Kaggle CUDA에서 `real_phase3_v2_6.h5` 재현 실행 후 결과 회수
6. Phase 5 ML 보정 학습/추론 checkpoint를 `pipelines/run_mode1.py --apply-correction` 훅에 연결
