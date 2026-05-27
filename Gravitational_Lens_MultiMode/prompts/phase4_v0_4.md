# Phase 4 v0.4 — fp16 SSIM NaN 수정 + 물리-validity-only 카탈로그

> v0.3.1까지의 결론: 학습 NaN의 진짜 원인은 fp16/autocast 하의 **SSIM 손실**이지
> 입력 크기/카탈로그 필터가 아니다(AMP off 시 NaN 0). 한편 v0.1/v0.2의 입력측 tail filter는
> "fp16 안정화"를 명분으로 도입된 **오진단 산물**이며, 실효는 selection bias뿐이었다.
> → SSIM을 fp32로 고치면 tail filter가 불필요 → v0.4는 물리 validity only로 train↔eval 분포를
> 일치시켜 selection bias를 구조적으로 제거한다.

## 진단 근거 (확정)
- v0.2: NaN 0, 그러나 leak(unfiltered/filtered RMSE 3.38, filtered r 0.65).
- v0.3: H0-중립 quota로 카탈로그 bias는 해결했으나 epoch1부터 NaN.
- v0.3.1: v0.2 입력 tail cap 복원(입력은 v0.2 안전범위 내) → **여전히 NaN**.
- v0.3.1 AMP off 재실행: NaN 0(50ep) → NaN은 fp16 전용. acceptance는 또 불합격(ratio 3.26 leak).
- v0.3 history: mode1/2/3_task·mode1_cal 유한, **ssim=nan** → 발원지는 `_ssim_loss`(fp16 분모 underflow + 분산 음수).

## 구현 (완료)
1. `ml/training/losses.py`: `_ssim_loss`/`_gaussian_nll`를 `torch.autocast(enabled=False)` fp32 블록으로
   강제. SSIM 분산 `clamp_min(0)`, NLL `2*log_sigma` `[-30,30]` 클램프 + var floor 1e-8.
   회귀테스트 `tests/test_losses_amp_safe.py` (degenerate 입력 fp16 finite + 정상입력) 4 passed.
2. `ml/data/error_catalog.py`: `validity_filter="v0_4"` — 물리 validity only
   (root/finite/`dt>0`/`|mu|<0.98`/sep≥0.1). label·tail cap·H0 quota 전부 제거(비-stratified 수집).
3. `data/mock/phase4_v0_4.h5`(n=500,seed42), `phase4_v0_4_eval_unfiltered.h5`(n=200,off),
   `target_scaler_phase4_v0_4.pkl`(mode1 mean/scale 29.64/13.71).
4. `scripts/phase4_v0_4_round.py`(v0.3.1 round 복제, 모델/입력/loss/optimizer/batch/AMP 동일).
5. `data/logs/phase4_v0_4_equivalence.json`(forward diff 1.043e-07, Welch p 0.874, passed).

## 핵심 결과 — train↔eval 분포 정합
| | train(v0_4) | unfiltered eval(off) |
|---|---|---|
| correction mean / std / max | 29.06 / 13.68 / 69.34 | 30.38 / 13.52 / 69.34 |
| H0 KS vs U[60,80] p | 0.025 | 0.029 |

v0.2 filtered correction(~17)의 truncation이 사라져 큰 correction tail이 학습에 포함됨 →
selection bias 구조적 제거 기대.

## Kaggle 학습 명령
```bash
# (M2) 업로드
python scripts/sync_to_kaggle.py --round phase4_v0_4 --init-dataset --slug lens-phase4-v0-4 --execute
# (Kaggle CUDA) sanity → full  (AMP on; SSIM fp32 수정으로 NaN 없어야 함)
python scripts/phase4_v0_4_round.py --phase train --epochs 2 --bootstrap-n 0
python scripts/phase4_v0_4_round.py --phase train \
  --equivalence-from /kaggle/input/lens-phase4-v0-4/phase4_v0_4_equivalence.json \
  --device cuda --workers 0 --epochs 50 --bootstrap-n 1000
```
기대: 2ep sanity `nan_detected=false`(SSIM fp32 효과). full run에서 unfiltered/filtered RMSE 비율이
1에 수렴(<=2.5), filtered r>=0.85.
