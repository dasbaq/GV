# Phase 4 v0.3.1 — input-tail validity filter (bias↔NaN 양립)

> Codex 실행용 프롬프트. v0.3은 selection bias를 고쳤으나(filtered H0 KS p 0.984)
> v0.2의 입력측 수치안정 tail gate까지 제거해 학습이 epoch1부터 결정적 NaN으로 붕괴했다.
> v0.3.1은 v0.3의 H0-중립 stratified quota를 유지하면서 입력측 tail gate만 v0.2 임계로 복원한다.

```text
역할: 너는 Gravitational_Lens_MultiMode 프로젝트의 ML 카탈로그 작업자다.
실행 환경: M2 로컬 (카탈로그/필터 설계·분석 전용. GPU 재학습은 하지 않는다).
작업 디렉토리: /Users/donghyun/Desktop/C_gravitational_lens/Gravitational_Lens_MultiMode

세션 규칙(필수):
- 시작 시 STATUS.md 전체 + CHANGELOG.md 상위 5개 + RUNBOOK.md를 읽는다.
- 정식 ML 스택은 ml/ 다. src_py/ml/ 는 수정/사용 금지.
- 상수는 config/physics.yaml, ML 하이퍼는 config/ml.yaml 에서만. 하드코딩 금지.
- 모델 구조/입력차원/loss/optimizer/batch/AMP 정책은 acceptance 없이 변경 금지.
  (이번 작업은 카탈로그 validity filter만 다룬다. AMP/loss 수정은 '제안'만, 실행 금지.)
- correction 부호 true - approx, corrected H0 = H0_approx + correction.
- .h5/.pkl/.pt 및 data/mock·checkpoints·logs·runs 는 git 커밋 금지.
- 손대기 전 대상 파일을 먼저 읽어 확인. 완료 후에만 CHANGELOG/STATUS/DECISIONS 갱신.

배경(확정 사실):
- v0.2: NaN 0이지만 selection-bias leak 발동(unfiltered/filtered RMSE 3.38, filtered r 0.65).
- v0.3: H0-중립 stratified quota로 bias 해결(filtered H0 KS p 0.984, 분포 정합 양호).
  그러나 학습이 epoch1부터 NaN(train nan_batches 25→42, epoch2 val까지 nan)으로 붕괴해
  acceptance 불가. 원인: v0.3이 label 의존 gate뿐 아니라 v0.2의 입력측 수치안정 tail gate
  (max|F_joint|<=3.408, I_obs.sum<=77.79, dt_approx<=444.7, |mu|<=0.9699, dphi_ratio band,
  separation>=0.6598)까지 함께 제거 → 극단 입력값 복귀로 AMP/fp16 overflow.

이번 작업: v0.3의 H0-중립 quota 설계는 유지하고, NaN을 유발한 '입력측' 안정화 컷만 복원한
v0.3.1 카탈로그를 만든다. label 의존 컷은 계속 제외해 bias를 막는다.

작업:
1. ml/data/error_catalog.py 의 v0_3 validity filter 구현과 v0_2 tail gate 정의를 읽어,
   각 컷을 (A) label 의존[H0_approx, correction] vs (B) 입력/관측측[F_joint, I_obs, dt_approx,
   mu, dphi_ratio, separation]으로 분류한다. 분류 근거를 표로 기록.
2. validity_filter="v0_3_1" 추가:
   - (A) label 의존 gate는 계속 제외(H0-중립 유지).
   - (B) 입력측 tail gate는 v0.2 임계로 복원하되, 각 컷이 H0 주변분포를 왜곡하지 않는지
     KS p로 검증(복원 후에도 filtered H0 KS vs U[60,80] p가 유의하게 크게 유지되어야 함).
   - H0 10-bin stratified quota는 v0.3와 동일하게 유지.
3. dphi_ratio band는 correction과 상관이 있을 수 있으니, 복원 시 H0/correction 왜곡 기여를
   별도로 측정해 bias를 다시 만들면 band를 넓히거나 제외하는 선택을 데이터로 결정한다.
4. v0.3.1 카탈로그(.h5, n=500, seed=42)와 unfiltered eval, label/floor/selection_bias json,
   target_scaler를 생성한다. NaN 안정성 사전점검: filtered split의 max|F_joint|, I_obs.sum,
   dt_approx, |mu|, correction의 max/percentile이 v0.2 안전범위 안인지 확인.

완료 기준:
- 컷 분류표(A/B) + v0.3.1 복원 후 filtered H0 KS p(>v0.3 수준 유지) + 입력 tail 통계가
  v0.2 안전범위 내임을 보고.
- filtered/unfiltered 분포 정합(H0·correction·dphi_ratio·mu·separation KS p)이 v0.3 대비
  유지/개선인지 표로 제시.
- scripts/phase4_v0_3_1_round.py 를 v0.3 round에서 artifact 이름만 바꿔 복제(모델/입력/loss/
  optimizer/batch/AMP 동일). selection-bias acceptance ratio<=2.5, coverage [0.62,0.78] 사전선언.
- NaN 재발 위험이 입력 컷만으로 안 잡히면, AMP/loss 측 수치 가드(예: F_joint 정규화, NLL var floor)를
  'acceptance 필요 제안'으로 STATUS/DECISIONS에 분리 기재(실행 금지).
- CHANGELOG 1줄 + STATUS '다음 작업'/'알려진 문제'에 v0.3 NaN 붕괴 원인과 v0.3.1 방향 기록.
  bias↔NaN 트레이드오프를 입력/label 컷 분리로 푼 결정을 DECISIONS.md에 1항목.
출력: 컷 분류표 + KS p + NaN 안전범위 점검 + 다음 Kaggle 재학습 명령(--phase train).
```

## 실행 결과 (2026-05-22 완료)

- `validity_filter="v0_3_1"` 구현. v0.3 H0 10-bin stratified quota 유지 + label gate 제외 +
  v0.2 입력측 tail gate 복원.
- `data/mock/phase4_v0_3_1.h5` (n=500, seed=42). filtered H0 KS vs U[60,80] p `0.999993`.
- NaN 안전범위 precheck 통과: `max|F_joint|=3.347<=3.408`, `I_obs.sum=69.03<=77.79`,
  `dt_approx=443.89<=444.7`, `|mu|max=0.969895<=0.9699`, `dphi_ratio=[0.58875,0.92007]`,
  separation min `0.66384>=0.6598`, correction max `32.261<=32.27`.
- `scripts/phase4_v0_3_1_round.py`: v0.3 round 복제(모델/입력/loss/optimizer/batch/AMP 동일),
  acceptance ratio `<=2.5`, coverage `[0.62,0.78]` 사전선언.
- 잔여 risk: filtered/unfiltered correction·dphi_ratio·separation KS p `0.0`로 dphi band가
  support를 좁힘. H0 uniformity는 유지되나 차기 라운드에서 재검토 대상.

## Kaggle 학습 명령

```bash
# (M2) 업로드
python scripts/sync_to_kaggle.py --round phase4_v0_3_1 --init-dataset --slug lens-phase4-v0-3-1 --execute
# (Kaggle CUDA) sanity → full
python scripts/phase4_v0_3_1_round.py --phase train --epochs 2 --bootstrap-n 0
python scripts/phase4_v0_3_1_round.py --phase train \
  --equivalence-from /kaggle/input/lens-phase4-v0-3-1/phase4_v0_3_1_equivalence.json \
  --device cuda --workers 0 --epochs 50 --bootstrap-n 1000
```
