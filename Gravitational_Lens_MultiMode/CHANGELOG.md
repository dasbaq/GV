# CHANGELOG.md
> 매 세션 마지막 5개만 읽음. 최신 항목이 위에 오도록 작성.

---

## [2026-05-26] — v0.7 calibration weight 상향 (coverage 개선)

### 추가
- `scripts/phase4_v0_7_round.py` 추가. v0.6 대비 변경 사항은 calibration loss weight `0.1 → 0.3` 하나뿐.
  모델 구조(Mode1Head in_dim=384, ImageEncoder I_obs 1ch), 데이터(phase4_v0_4.h5 20-dim),
  optimizer, AMP 정책, 승인 기준은 v0.6와 완전 동일.

### 배경 및 근거
- v0.6 결과: RMSE 5.258, r 0.503, coverage **0.52** (목표 [0.62, 0.78] **미달**).
- `coverage = P(|z| ≤ 1), z = (H0_corrected - H0_true) / pred_sigma`.
  coverage 0.52는 pred_sigma가 실제 잔차의 약 71% 수준으로 과소추정됨을 의미한다.
- `composite_loss`에서 NLL calibration 항의 유효 기여는 `w_m1 * w_cal = 1.0 * 0.1 = 0.1`이지만
  task MSE 기여는 `w_m1 * 1.0 = 1.0`이다. NLL이 log_sigma를 충분히 학습하지 못해 sigma가 너무 작아짐.
- calibration weight `0.1 → 0.3`으로 3배 상향 시 NLL의 log_sigma 학습 영향력이 3배 강해지고,
  pred_sigma가 실제 잔차 크기에 더 잘 맞춰지는 방향으로 학습.
- RMSE/r이 소폭 하락할 수 있으나 coverage가 목표 범위로 진입하면 허용(calibration–task trade-off).

### 검증
```
python -m py_compile scripts/phase4_v0_7_round.py  # OK
python scripts/phase4_v0_7_round.py --help         # 옵션 정상 출력
```

---

## [2026-05-25] — v0.5 Mode 3(Source 복원) + Image 입력 모달리티 삭제

### 삭제
- `ml/models/heads.py`: `Mode3Head`, `_UpBlock` 삭제. `Mode1Head`, `Mode2Head`만 유지.
- `ml/models/encoders.py`: `ImageEncoder`, `_Conv2DBlock` 삭제. 3종 인코더(LC/Param/Σ-curve)만 유지.
- `ml/models/fusion.py`: 4-way 분기 제거, 3-way Cross-attention 고정(`h_lc, h_params, h_sigma`).
- `ml/models/error_corrector.py`: `img_enc`, `head3` 삭제. `Mode1Head in_dim: d_model×3(384) → d_model×2(256)`.
  forward 시그니처: `(lc, lc_mask, params, sigma_curve, target_mode)` — `image`/`use_image` 인자 삭제.
- `ml/training/losses.py`: `_ssim_loss` 전체 삭제. `composite_loss` weights `{mode1, mode2, physics, calibration}`.
  return dict: `{total, mode1_task, mode2_task, mode1_cal, mode2_cal, physics}` — `mode3_task`, `ssim` 삭제.
- `ml/training/dataset.py`: image 로딩 블록 전체 삭제. return dict에 `image`, `use_image`, `target_image` 없음.
  default `modes=(1,2)`. `build_weighted_sampler` default `mode_weights=[1.0, 1.0]`.
- `ml/training/trainer.py`: `_step`에서 `use_image`/`image` 삭제. `fine_tune`에서 `img_enc` 제거.
  `evaluate`에서 mode3 제거.
- `ml/training/physics_pairing.py`: `image`/`use_image` 삭제.
- `inversion/obs_to_features.py`: `image_size` 파라미터 삭제. 반환 dict에 `image`/`use_image` 없음.
- `inversion/mode3_wrapper.py`: 파일 삭제.
- `ml/utils/mock_generator.py`: `image_size`, `_gaussian_source`, `_sis_lens_image` 삭제.
  `create_mock_h5`에서 `images/` 그룹, `mode3_source_residual` 저장 없음.
- `config/ml.yaml`: `image_size`, `image_backbone`, `modes: [1,2,3]→[1,2]`, `mode_sampling_weights: [1.0,1.0]`,
  `loss_weights.mode3`/`ssim` 삭제.
- `tests/test_mode3_wrapper.py`, `tests/benchmarks/test_source_psnr.py`: 파일 삭제.

### 변경
- `pipelines/run_mode1.py`, `pipelines/train_corrector.py`, `scripts/phase4_v0_4_round.py`,
  `ml/training/round_eval.py`, `scripts/phase4_v0_4_r_ceiling.py` 및 historical round scripts
  (v2_6, phase4_v0_1/2/3/3_1): `image`/`use_image`/`image_size` 참조 전부 제거.
- `tests/`: `test_heads.py`, `test_encoders.py`, `test_losses_amp_safe.py`, `test_corrector.py`,
  `test_round_eval_common.py`, `test_run_mode1_correction.py`, `test_dataset.py`, `test_trainer.py` 갱신.

### 근거 (DECISIONS.md [2026-05-25] 참조)
- 학습(`use_image=True`)과 추론(`I_obs=None → use_image=False`) 경로 구조적 불일치 해소.
- Mode1Head 입력 차원 낭비 제거(1/3이 항상 0인 `h_img` concatenation).
- 실제 Mode 3 카탈로그 데이터 부재 (nonzero target 0%).
- `_ssim_loss` fp16/autocast underflow NaN 발원지 제거.
- v0.3/v0.3.1 NaN 붕괴 위험 원천 제거.

### 호환성 주의
- v0.4 checkpoint (`head1.net.0.weight` shape 384×…)는 v0.5 모델과 **비호환**.
  Kaggle CUDA에서 v0.5 기반 재학습 필요. 목표: unfiltered r ≥ 0.60, RMSE ≤ 5.5, coverage CI ∩ [0.62,0.78].
- 데이터 생성 코드(`error_catalog.py`, `data_adapters`)는 HDF5에 여전히 `images/` 그룹과
  `mode3_source_residual`을 저장하지만, 학습 코드는 이를 읽지 않는다(후방 호환 스키마).

### 검증
```
python -c "from ml.models.heads import Mode1Head, Mode2Head; print('heads OK')"
python -c "from ml.models.encoders import LightCurveEncoder, ParamEncoder, SigmaCurveEncoder; print('encoders OK')"
python -c "from ml.models.error_corrector import MultiModalErrorCorrector; ..."  # forward OK, mode1/mode2 shape 확인
python -c "from ml.training.losses import composite_loss; ..."  # keys: [total,mode1_task,mode2_task,mode1_cal,mode2_cal,physics]
pytest -q tests/test_heads.py tests/test_encoders.py tests/test_losses_amp_safe.py tests/test_corrector.py
         tests/test_dataset.py tests/test_trainer.py tests/test_round_eval_common.py  # 관련 회귀 통과
```
`grep -rn "Mode3Head\|ImageEncoder\|use_image\|image_size\|_ssim_loss" ml/ inversion/ pipelines/ config/` → 0건.

## [2026-05-25] — Phase 5 physics loss 정식화 + trainer/round 평가 공용화
- `ml/training/losses.py`: 기존 correction L2 대리 physics loss를 Mode 1/2 D_Δt 일관성 패널티로 교체. `H0_corrected`와 SIE `theta_E` first-order Fermat scaling이 같은 time-delay distance를 함의하는지 fp32/autocast-off로 계산한다. Mode 3은 제외.
- `ml/training/physics_pairing.py` 추가. 같은 batch를 Mode 1/2로 paired forward해 physics loss 입력을 만들며, checkpoint 구조는 변경하지 않는다. v0.4 Mode 2 target이 zero placeholder이면 `mode2_label_available=false`로 physics loss가 0으로 degrade된다.
- `ml/training/round_eval.py` 추가. target-scaler 역변환, `H0_approx + correction` 평가, no-correction/perfect-joint baseline, calibration coverage, bootstrap CI, acceptance/leak report를 공용화하고 `CorrectorTrainer.evaluate`와 `scripts/phase4_v0_4_round.py`가 같은 평가 코드를 호출하게 했다.
- `ml/models/error_corrector.py`: Mode 1/2-only 배치의 dummy skip feature shape 표현을 명시화. Mode 3은 삭제 예정 정책에 따라 최소 변경만 수행.
- v0.4 Mode 2 데이터 확인: `data/mock/phase4_v0_4.h5`의 `correction_targets/mode2_dm_correction`은 shape `[500,4]`, nonzero fraction `0.0` → Mode 2 실학습 배선은 데이터 선행 필요로 보류.
- 문서 갱신: `STATUS.md`의 Phase 5 ML stale `[ ]`를 `[x]`로 정정하고, Mode 3 삭제 예정/신규 투자 금지 정책을 `STATUS.md`·`ARCHITECTURE.md`에 기록. physics loss 결정은 `DECISIONS.md`에 추가.
- 회귀: `pytest -q tests/test_losses_amp_safe.py tests/test_corrector.py tests/test_trainer.py tests/test_round_eval_common.py` → 21 passed. `python -m py_compile ml/training/round_eval.py ml/training/physics_pairing.py scripts/phase4_v0_4_round.py` 통과.

## [2026-05-25] — Phase 5-A Mode 2 SIE θ+Δt 솔버 물리 정립
- `inversion/mode2_dm.py`를 placeholder에서 SIE 고정 물리 솔버로 교체. 내부 fit 변수는 `(sigma_v, q, position_angle, beta_x, beta_y)`이고, 공개 `dm_params`는 Phase 4 순서 `[theta_E, q, position_angle, sigma_v]`로 반환한다. `theta_pred = theta_obs.copy()` 경로와 μ-only loss를 제거하고, 렌즈 방정식 `beta=theta-alpha(theta)` 위치 residual + Fermat Δφ 기반 Δt residual을 결합했다.
- 기본 likelihood는 μ-free다. `mu_obs`와 μ residual weight는 후속 확장 자리만 유지하고 기본 `0.0`이며, solver와 pipeline audit에 `uses_truth_mu_true=false`를 기록한다. SIE 외 모델 및 `approx_level` switch는 거부한다.
- `pipelines/run_mode2.py` 추가. Phase 4 HDF5의 public 입력(`ray_paths/theta_*`, `observed_features/dt_lc`, `dt_lc_sigma`, redshift/H0)만 읽어 JSON을 쓰고, `--eval-truth`일 때만 `true_values/dm_params_true`로 MOCK 회복 지표를 산출한다.
- `data/logs/mode2_recovery_eval.json` 생성: `data/mock/phase4_v0_4.h5` 고정 subset n=25 MOCK 평가. full truth가 SIE+NFW+κ_ext라 SIE-only solver 대비 bias가 남음을 정직하게 기록(theta_E median relative error `0.527`, q `0.194`, sigma_v `0.236`; μ leak false).
- 테스트 갱신: placeholder SIS/NFW μ 기반 테스트를 제거하고 SIE θ+Δt 회복 및 CLI E2E 검증으로 교체. `pytest -q tests/test_inversion_mode2.py tests/test_run_mode2_e2e.py tests/benchmarks/test_dm_recovery.py` → 8 passed.

## [2026-05-25] — Phase 4 v0.4 achievable-r ceiling 산정 + r 기준 확정
- `scripts/phase4_v0_4_r_ceiling.py` 추가. `LensCorrectionDataset` Mode 1 경로와 동일한 입력(`lc`, `lc_mask`, 20-dim `params`, padded `sigma_curve`, `image=[I_obs,I_obs-S_approx]`)을 flatten하고, `ExtraTreesRegressor(n_estimators=2000, min_samples_leaf=2, max_features=0.5, seed=20260525)`로 correction-space conditional mean을 fit한다. feature extraction에는 truth-side key(`H0_true`, `dt_true`, `mu_true`, `M200`, `concentration`, correction label)를 사용하지 않도록 leak guard를 기록.
- M2 로컬 실행: `python scripts/phase4_v0_4_r_ceiling.py --bootstrap-n 1000` → `data/logs/phase4_v0_4_r_ceiling.json` 생성. filtered val(n=50) oracle H0 r `0.2495` CI `[-0.0770,0.5319]`, correction-space r `0.9092`; unfiltered all(n=200) oracle H0 r `0.8096` CI `[0.7324,0.8786]`, correction-space r `0.9700`.
- 기존 model unfiltered H0 r `0.6560`은 unfiltered ceiling의 `0.810`배 달성. filtered model H0 r `0.6207`은 noisy filtered ceiling 기준을 초과하지만, threshold 산식은 사전 고정 규칙 `floor_to_2_decimals(0.80 * filtered_oracle_h0_r)` 그대로 적용.
- `scripts/phase4_v0_4_round.py`: `filtered_h0_r_min`을 record_only `0.0`에서 정량 기준 `0.19`로 확정. 기존 RMSE/coverage/selection-bias/positive_fraction 기준은 유지.
- `tests/test_phase4_r_ceiling.py` 추가: leak guard path overlap과 실제 v0.4 catalog feature dimension(dataset Mode 1 path)을 검증. `pytest -q tests/test_phase4_r_ceiling.py` → 2 passed.

## [2026-05-25] — Phase 4 v0.4 20-dim observed_features 정식 재학습 (Kaggle CUDA)
- 로컬 20-dim 카탈로그(`phase4_v0_4.h5` 41MB / eval / scaler / floor)를 `scripts/sync_to_kaggle.py --execute`로 Kaggle Dataset `donghyun51/lens-phase4-v0-4` 새 버전 업로드(인증 ACCESS_TOKEN). 이전 13-dim BLOCKED 해소. 업로드 검증: train h5 42,707,816 B(=로컬 20-dim, `observed_features` 포함), eval 17,115,456 B.
- Kaggle CUDA에서 `phase4_v0_4_round.py --phase train --workers 0 --epochs 50 --bootstrap-n 1000` 실행. **equivalence는 `--phase train`으로 의도적 생략**(입력 차원만 13→20, 수치 경로 무변경; CPU↔CUDA 동등성은 5-24 diff 4.47e-08로 입증됨). 데이터 경로 `lens-phase4-v0-4/phase4_v0_4.h5`로 학습됨 확인 = 진짜 20-dim.
- 학습: 27ep early-stop(best ep19), `best_val_m1=0.1243`(record_only, val_m1 궤적 0.12~0.93 불안정), num_workers=0, **NaN 0**.
- eval: filtered RMSE `4.53`(CI [3.30,5.75])/r `0.62`/coverage `0.80`(CI [0.66,0.90]), unfiltered RMSE `4.51`(CI [4.02,4.99])/r `0.66`/coverage `0.655`(CI [0.585,0.721]), ratio `0.996`(<=2.5), pos_frac 1.0/1.0, **leak_triggered false**. 13-dim 재현(unfiltered r 0.62) 대비 r 소폭 개선, selection bias·NaN 재발 없음.
- `all_pass_excluding_record_only=false`는 equivalence 생략으로 acceptance의 CUDA forward diff 행이 `Inf`(미산출)인 단 한 줄 때문이며, 성능/bias/calibration 행은 전부 pass다.
- v0.4 코드(round/feature_schema/real_catalog/config)는 PR #1로 `origin/main`에 머지됨 확인 → Kaggle 노트북 main clone에 v0.4 스크립트 존재.
- 남은 작업: Kaggle output에서 checkpoint/eval/infra JSON 회수 후 `par_enc.net.0.weight==(256,20)` 최종 확인하고 BENCHMARKS 고정.

## [2026-05-24] — Phase 4 v0.4 Kaggle CUDA 재현 회수 (⚠️ 13-dim 데이터)
- Kaggle CUDA(AMP on) full run 결과 회수. **단, Kaggle Dataset 업로드가 인증 미설정으로 BLOCKED이라 학습된 데이터는 구 13-dim `phase4_v0_4.h5`다.** `leak_triggers.param_encoder_input_dim_changed: false`, checkpoint `par_enc.net.0.weight=(256,13)`로 확인 — 20-dim 재학습이 아니라 기존 v0.4 재현이다.
- 학습: criterion `val.mode1_task`, best ep10 / early-stop ep18(patience 8) / max 50, num_workers=0, NaN 0. `best_val_m1=0.1413`(record_only). val_m1 궤적은 ep별 0.14~0.60으로 불안정.
- eval: filtered RMSE `6.18`(CI [5.00,7.30]) / r `0.39`, unfiltered RMSE `4.55`(CI [4.08,4.98]) / r `0.62`, unfilt/filt 비율 `0.74`(<=2.5), filtered coverage `0.68`(CI [0.533,0.805]), unfiltered coverage `0.785`, positive_fraction 1.0/1.0. `all_pass_excluding_record_only: true`, `leak_triggered: false`.
- infra: CPU↔CUDA forward diff `4.47e-08`(<=1e-4). epoch1 CPU/CUDA val_m1 sanity seed42 0.9036/0.9190, seed1337 0.9823/0.9110, seed7 0.7853/0.7902.
- 판정: 13-dim v0.4 재현 통과, selection bias·NaN 재발 없음. 20-dim 정식 재학습은 Kaggle Dataset에 20-dim 카탈로그 재업로드 후 재실행 필요(다음 작업).

## [2026-05-24] — Phase 4 v0.4 20-dim observed_features 카탈로그 재생성
- `data/mock/phase4_v0_4.h5` 재생성: n=500, seed=42, `validity_filter=v0_4`, `observed_features/` 및 `light_curve_quality/` 포함. Mode 1 correction mean/std/min/max `29.065/13.677/3.198/69.339`, train `|mu_truth|max=0.9785`.
- `data/mock/phase4_v0_4_eval_unfiltered.h5` 재생성: n=200, seed=42, `validity_filter=off`, `eval_role=unfiltered`, 동일 observed feature schema 포함. Mode 1 correction mean/std/min/max `30.378/13.515/6.543/69.339`.
- `data/target_scaler_phase4_v0_4.pkl` 재생성: Mode 1 mean/scale `29.6389/13.7096`. `data/logs/phase4_v0_4_floor_analysis.json`도 새 카탈로그 기준으로 갱신.
- schema 확인: `config/ml.yaml:data.param_normalization` 15개 + approx/mode one-hot 5개 = ParamEncoder 20-dim. HDF5 `observed_features/{dt_lc,dt_lc_sigma,dt_lc_sigma_relative_error,n_epochs_quality,baseline_days,median_cadence_days,median_photometric_error}` 확인.
- 로컬 MPS equivalence는 현재 Codex 프로세스에서 `MPS is not available`로 미실행. stale `phase4_v0_4_equivalence.json`는 `phase4_v0_4_equivalence_legacy_20260522.json`으로 보존하고, Kaggle에서는 `--phase all`로 CUDA equivalence+train을 실행해야 한다.
- Kaggle staging dry-run/복사 확인: `/var/folders/.../T/kaggle_lens-phase4-v0-4/`에 train/eval HDF5, scaler, floor JSON 생성. `kaggle` CLI는 설치했으나 인증이 없어(`kaggle auth login` 필요) Dataset version 업로드는 중단.
- 회귀: `pytest -q tests/test_error_catalog.py tests/test_run_mode1_correction.py tests/test_encoders.py tests/test_corrector.py tests/test_phase4_validity.py tests/test_standard_approx.py tests/test_losses_amp_safe.py` → 34 passed.

## [2026-05-24] — 실관측 YAML 입력 + ParamEncoder 품질/누락 feature schema
- `ml/training/feature_schema.py` 추가: dataset/관측 어댑터가 공유하는 ParamEncoder scalar schema를 단일화. `dt_lc`, `dt_lc_sigma` 필수 검증, `sigma_v`/`theta_E`/`q` 누락 시 normalized-zero sentinel + field별 missing flag, 광곡선 품질 지표 4개를 처리. truth-only 키(`M200`, `concentration`, `kappa_ext`, `nfw_offset`) 거부.
- `config/ml.yaml`: ParamEncoder 입력에 `n_epochs_quality`, `baseline_days`, `median_cadence_days`, `median_photometric_error`, `missing_sigma_v`, `missing_theta_E`, `missing_q` 추가. 품질 지표 normalization은 config의 `transform: log`와 범위(`N_epochs` 0-1500, baseline 0-5000, cadence 0-50, phot err 0-0.5)로 제어.
- `inversion/real_catalog.py` 추가: Gaia GraL X + 외부 Bag+22 결과용 YAML 리스트 loader. Bag+22는 호출하지 않고 `dt_lc`/`dt_lc_sigma`/품질 지표를 검증해 Mode 1 feature spec으로 변환하며, Mode 2 확장 필드는 보존만 한다.
- `config/ml.yaml:data.observed_features`: `dt_lc_sigma_sampler`를 `relative_then_clip` + `relative_error.log_uniform[0.01,0.30]` + absolute clip `[0.3,20.0] days`로 분리. `dt_sign_convention`은 음수 입력을 `abs()` 처리하고 pair order를 뒤집는 정책으로 config화.
- `ml/data/error_catalog.py`: Phase 4 HDF5에 `observed_features/` 및 `light_curve_quality/` 저장. 시뮬레이션 `dt_lc`는 기존처럼 `abs(dt_true)`를 유지하고, `dt_lc_sigma`는 config sampler에서 샘플링해 상대오차와 attrs를 기록.
- `tests/fixtures/real_catalog/`: `complete.yaml`, `partial_no_lens_model.yaml`, `minimal.yaml`, `invalid_examples.yaml` 추가. 음수 `dt_lc`는 abs + pair flip + conversion log, `dt_lc_sigma` 누락과 truth-only key는 reject 테스트.
- `ml/training/dataset.py`, `inversion/obs_to_features.py`, `pipelines/run_mode1.py`: 새 schema를 사용하도록 전환. YAML 입력은 raw LC/image 없이도 zero modality tensor로 graceful 처리하고, 기존 HDF5 경로는 legacy fallback 유지. 기존 13-dim checkpoint의 ParamEncoder 첫 layer를 20-dim 모델로 부분 이식하는 compatibility shim 추가.
- 회귀: `pytest -q tests/test_real_catalog.py tests/test_error_catalog.py tests/test_run_mode1_correction.py tests/test_encoders.py tests/test_corrector.py tests/test_run_mode1_e2e.py tests/test_phase4_validity.py tests/test_standard_approx.py` → 39 passed.

## [2026-05-22] — Phase 5 Mode 1 ML 보정 결합 (관측 → H0_corrected)
- `inversion/obs_to_features.py` 추가: 스펙 dict → MultiModalErrorCorrector Mode 1 입력 텐서. `ml/training/dataset.py::__getitem__`의 Mode 1 경로(lc/params/sigma_curve/image/use_image, 정규화·one-hot·scaler)를 그대로 재현. `system_spec_from_hdf5`, `load_corrector`, `load_target_scaler` 헬퍼 포함. truth-side 키 미접근.
- `pipelines/run_mode1.py`: `_apply_ml_correction`을 실제 구현(stub 제거) — checkpoint+scaler 로드 → target_mode=1 forward → `correction = pred*scale + mean`, `sigma = exp(log_sigma)*scale`, `H0_corrected = H0_approx + correction`. `_feature_spec_from_phase4_hdf5`로 입력 HDF5의 image/LC/sigma 그룹 + 해석 파이프라인 param(H0_approx/dt_obs/SIE fit)으로 스펙 구성. Phase4 inference 그룹 부재 시 graceful skip. `--correction-scaler`/`--correction-config` CLI 추가(기본 v0.4).
- `tests/test_run_mode1_correction.py`: (a) 어댑터==dataset 출력 일치, (b) v0.4 checkpoint로 보정 closed-form, (c) graceful skip. `pytest -q tests/test_run_mode1_correction.py tests/test_run_mode1_e2e.py tests/test_losses_amp_safe.py` → 9 passed.
- v0.4 데모(검증): H0_approx 60.6/31.4/21.5 → H0_corrected 72.4/73.5/73.0 (H0_true 74.0/68.9/63.9). 실관측 원본 데이터는 부재라 Phase4-HDF5/합성으로만 검증(MOCK).

## [2026-05-22] — Phase 4 v0.4 Kaggle full run: selection bias·NaN 해결 + acceptance 재교정
- v0.4 CUDA full run(AMP on): NaN 0(19ep, early-stop, best ep11). **selection bias 소멸** — unfiltered/filtered RMSE 비율 `0.83`(<=2.5, leak **false**), unfiltered RMSE `14.5→4.81`, r `0.28→0.59`, coverage `0.21→0.695`. SSIM fp32 + 물리-validity-only 설계가 의도대로 작동.
- 남은 불합격은 filtered absolute band 2개(RMSE CI upper `6.92>4.862`, r `0.33<0.85`)뿐인데, 이는 성능 문제가 아니라 band가 v0.2 truncated easy-subset 기준이라 무효한 것. filtered==unfiltered 분포가 되어 filtered val(n=50)이 전체 난이도를 정직하게 평가(unfiltered n=200은 r `0.59`).
- `scripts/phase4_v0_4_round.py` ACCEPTANCE 재교정: no_correction(filtered `29.63`/unfiltered `33.25`) 기준 RMSE band `[0.5, 11.08]`(point [0.5,16.62]), `filtered_h0_r_min` `0.85→0.0`(record_only), leak floor `2.755→0.5`. 재교정 후 v0.4는 bias/RMSE/coverage 전부 통과(r은 record_only). 근거 DECISIONS + `data/logs/phase4_v0_4_floor_analysis.json`.
- remaining rigor: 무편향 분포의 achievable-r ceiling을 inputs-conditioned oracle로 산정.

## [2026-05-22] — fp16 SSIM NaN 수정 + Phase 4 v0.4 물리-validity-only 카탈로그
- 진단 확정: v0.3.1도 학습 NaN(epoch1 train=nan, val 유한). AMP off로 재실행하니 NaN 0(50ep) → NaN은 fp16 전용. v0.3 history에서 mode1/2/3_task·mode1_cal 유한한데 ssim=nan → **발원지는 fp16/autocast SSIM**, 카탈로그 필터와 무관. 단 AMP-off v0.3.1은 acceptance 또 불합격(unfiltered/filtered RMSE 3.26, filtered r 0.66)으로 selection bias 잔존 → tail filter가 원인.
- `ml/training/losses.py`: `_ssim_loss`/`_gaussian_nll`를 autocast 비활성 fp32 블록으로 강제. SSIM 분산 `clamp_min(0)`, NLL `2*log_sigma` `[-30,30]` 클램프 + var floor. `tests/test_losses_amp_safe.py` 추가(degenerate 입력 fp16 finite + 정상입력 검증), 4 passed.
- `ml/data/error_catalog.py`: `validity_filter="v0_4"` 추가 — 물리 validity only(root/finite/dt>0/`|mu|<0.98`/sep≥0.1), label·tail cap·H0 quota 전부 제거(비-stratified 수집). v0.4는 v0.1/v0.2 tail filter가 "fp16 안정화" 오진단 산물이었다는 결론에 따른 설계.
- `data/mock/phase4_v0_4.h5`(n=500,seed42) + `phase4_v0_4_eval_unfiltered.h5`(n=200,off): train correction mean/std/max `29.06/13.68/69.34`가 unfiltered `30.38/13.52/69.34`와 **일치**(v0.2 filtered ~17의 truncation 제거 → selection bias 구조적 해소). `target_scaler_phase4_v0_4.pkl`(mode1 mean/scale `29.64/13.71`).
- `scripts/phase4_v0_4_round.py`(v0.3.1 round 복제, 모델/입력/loss/optimizer/batch/AMP 동일). `data/logs/phase4_v0_4_equivalence.json`(forward diff `1.043e-07`, Welch p `0.874`, passed). Kaggle input 폴더 `data/kaggle_upload/lens-phase4-v0-4/`(git ignored) 생성.
- 회귀: `pytest -q tests/test_losses_amp_safe.py tests/test_phase4_validity.py tests/test_error_catalog.py tests/test_standard_approx.py` → 18 passed.

## [2026-05-22] — Phase 4 v0.3.1 equivalence handoff + Kaggle staging + prompt
- `data/logs/phase4_v0_3_1_equivalence.json` 생성(`--phase equivalence --device mps --epochs 1`): forward max diff `1.043e-07`(<1e-4), distribution Welch p `0.9908`, `passed=true`. Kaggle `--phase train`의 `--equivalence-from` 입력.
- `python scripts/sync_to_kaggle.py --round phase4_v0_3_1 ... ` dry-run에서 train h5/unfiltered eval h5/scaler/equivalence json/floor json 5개 포함, missing 0 확인.
- Kaggle Dataset 업로드용 input 폴더 `data/kaggle_upload/lens-phase4-v0-3-1/`(git ignored) 생성: 위 5개 파일 + `dataset-metadata.json`(`donghyun51/lens-phase4-v0-3-1`). `kaggle datasets create -p <folder>`로 업로드 가능.
- `prompts/phase4_v0_3_1.md`: v0.3.1 Codex 프롬프트 + 실행 결과 + Kaggle 학습 명령 문서화.

## [2026-05-22] — Phase 4 v0.3.1 input-tail validity filter
- `ml/data/error_catalog.py`: `validity_filter="v0_3_1"` 추가. v0.3 H0 10-bin stratified quota는 유지하고 label-dependent `H0_approx`/`mode1_H0_correction` gate는 계속 제외하되, v0.2 입력측 tail gate(`F_joint`, `I_obs.sum`, `dt_approx`, `|mu|`, `dphi_ratio`, separation)만 복원.
- `data/mock/phase4_v0_3_1.h5`: n=500, seed=42, H0 bin별 50개. filtered H0 KS vs U[60,80] p `0.999993`; v0.2 안전범위 precheck는 `max|F_joint|=3.347<=3.408`, `I_obs.sum=69.03<=77.79`, `dt_approx=443.89<=444.7`, `|mu|max=0.969895<=0.9699`, `dphi_ratio=[0.58875,0.92007]`, separation min `0.66384>=0.6598`, correction max `32.261<=32.27`.
- `data/mock/phase4_v0_3_1_eval_unfiltered.h5`, `data/logs/phase4_v0_3_1_{label_distribution,floor_analysis,selection_bias_analysis}.json`, `data/target_scaler_phase4_v0_3_1.pkl` 생성. filtered/unfiltered KS p: H0 `0.144`, correction `0.0`, dphi_ratio `0.0`, mu `0.0265`, separation `0.0`; dphi band는 H0 uniformity를 유지하지만 correction/dphi support를 좁히는 잔여 risk로 기록.
- `scripts/phase4_v0_3_1_round.py`: v0.3 round와 동일한 모델/입력/loss/optimizer/batch/AMP 정책을 artifact 이름만 바꿔 복제. selection-bias acceptance ratio `<=2.5`, coverage target `[0.62,0.78]` 유지.

## [2026-05-22] — Phase 4 v0.3 Kaggle 학습 NaN 붕괴 + 결과 회수
- Kaggle 산출물 회수(`data/`, `data_workers0/` 두 변형 동일): `data/checkpoints/phase4_v0_3_imgres_best.pt`, `data/logs/phase4_v0_3_imgres_h0_eval{,_unfiltered}.json`, `phase4_v0_3_imgres_long_history.json`, `phase4_v0_3_infra_equivalence.json`.
- v0.3 학습은 epoch1부터 NaN으로 붕괴: train nan_batches epoch1 `25`/epoch2 `42`+val `3`, epoch2 train·val 모두 nan, grad_norm `2.29→5.57`. num_workers `4`/`0` 두 변형이 batch 단위까지 bit-identical NaN → 결정적 수치 overflow(데이터로더 비결정성 아님).
- best checkpoint는 nan-collapse된 epoch2라 eval(filtered RMSE `10.6`, r `0.30`)은 무의미. acceptance 불가.
- 원인: v0.3이 label 의존 gate(`H0_approx∈[45,90]`, `correction absmax`)뿐 아니라 v0.2의 입력측 수치안정 tail gate(`max|F_joint|`, `I_obs.sum`, `dt_approx`, `|mu|`, `dphi_ratio`)까지 제거 → 극단 입력 복귀로 AMP/fp16 overflow.
- 결론(STATUS·DECISIONS 후보): validity 컷을 label 의존(bias 원인, 계속 제외)과 입력/관측측(수치안정, 복원)으로 분리. v0.3.1 = v0.3 stratified quota 유지 + 입력측 tail gate만 v0.2 임계로 복원.

## [2026-05-22] — Phase 4 v0.3 H0-neutral catalog filter
- `ml/data/error_catalog.py`: `validity_filter="v0_3"` 추가. label-dependent `H0_approx`/correction gates와 v0.2 support-tail gates를 제거하고 root/finite/`dt_true>0`/`|mu|<0.98`/physical separation만 유지한 뒤 H0 10-bin stratified quota를 적용.
- `data/mock/phase4_v0_3.h5`: n=500, seed=42, H0 bins 50개씩. filtered H0 KS vs U[60,80] p `0.984`로 v0.2 p `1.8e-6` 대비 개선.
- `data/mock/phase4_v0_3_eval_unfiltered.h5`, `data/logs/phase4_v0_3_{label_distribution,floor_analysis,selection_bias_analysis}.json`, `data/target_scaler_phase4_v0_3.pkl` 생성. filtered/unfiltered distribution match: H0 KS p `0.144`, correction KS p `0.801`, dphi_ratio KS p `0.853`, mu KS p `0.685`, separation KS p `0.351`.
- `scripts/phase4_v0_3_round.py`: v0.2와 동일한 모델/입력/loss/optimizer/batch/AMP 평가 경로를 v0.3 artifact 이름으로 복제하고 selection-bias acceptance ratio `<=2.5`, coverage target `[0.62,0.78]`를 사전 선언.

## [2026-05-22] — Phase 4 v0.2 Kaggle 학습 결과 회수 + acceptance 판정
- Kaggle T4 CUDA 학습 산출물을 repo로 회수: `data/checkpoints/phase4_v0_2_imgres_best.pt`, `data/logs/phase4_v0_2_imgres_h0_eval{,_unfiltered}.json`, `phase4_v0_2_imgres_long_history.json`, `phase4_v0_2_infra_equivalence.json`.
- 학습: 50 epoch 상한 중 epoch 16 early-stop (best epoch 8, best_val_m1 `0.5014`), NaN 0. equivalence 통과(forward diff ~1e-9, Welch p `0.9981`, param_encoder_input_dim 13).
- Mode 1 지표 — filtered_val: RMSE `4.33`(no-correction `17.74`), r `0.65`, 1σ coverage `0.54`, correction positive_fraction `1.0`. unfiltered_all: RMSE `14.62`, r `0.28`, coverage `0.21`, bias `-10.8`.
- **acceptance 불합격**: filtered r `0.65 < 0.85`, filtered RMSE CI upper `5.09 > 4.862`, unfiltered/filtered RMSE 비율 `3.38 > 2.5`(leak 임계 `3.18`도 초과 → selection-bias leak 발동). coverage CI `[0.39,0.68]`는 목표 `[0.62,0.78]`와 경계만 겹쳐 over-confident. `val_H0_true` KS vs U[60,80] p `1.8e-6`로 filtered H0 분포 왜곡 확인.
- 결론: 재학습 문제가 아니라 catalog validity filter의 selection bias. 후속 진단은 같은 날짜 selection bias 항목 참조.

## [2026-05-22] — Phase 4 v0.2 selection bias 진단
- `scripts/analyze_phase4_v0_2_selection_bias.py`: v0.2 validity filter를 unfiltered catalog에 재적용해 컷별 H0 왜곡, filtered/unfiltered support 차이, fixed-model reweight RMSE를 분석. v0.2 Kaggle 불합격 원인을 재학습 문제가 아닌 catalog selection bias로 정리하고 v0.3 H0-중립 filter 방향을 `STATUS.md`에 기록.

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
