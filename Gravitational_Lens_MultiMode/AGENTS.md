@RTK.md
## 세션 시작 — 파일 읽기 순서

1. STATUS.md 전체 읽기 → 현재 Phase와 다음 작업 파악
2. CHANGELOG.md 위에서 5개 항목만 읽기 → 최근 변경 파악
3. 작업 시작

아래 상황에서는 추가로 읽을 것:
- 요청이 디렉토리·스키마·물리 공식과 관련됨 → ARCHITECTURE.md 읽기
- 요청이 이전 설계 결정과 충돌할 수 있음 → DECISIONS.md 읽기
- 벤치마크 결과 확인 필요 → BENCHMARKS.md 읽기

---

## 작업 규칙

**충돌 감지**
- 요청이 STATUS.md의 현재 Phase와 충돌 → 작업 전 먼저 알릴 것
- 구조 변경 필요 → ARCHITECTURE.md 읽고 확인 후 작업
- ML 라운드 진입 요청이면 먼저 실행 환경을 명시:
  - 시뮬레이션/카탈로그/scaler/floor 분석은 M2 로컬
  - equivalence 검증은 M2 로컬에서 `--phase equivalence`
  - GPU 학습 + bootstrap 평가는 Kaggle CUDA
    (`--phase train`, smoke run은 acceptance/leak 판정 skip)
  - checkpoint/log 회수 및 문서화는 M2 로컬
  환경이 맞지 않으면 실행 대신 필요한 명령과 산출물 경로를 보고할 것.

**코드 작성**
- 물리 상수 → config/physics.yaml 에서만
- Δt_try / μ_try 스윕 → for 루프 금지, 반드시 벡터화 (2D 그리드 일괄 처리)
- |μ| < 1 수렴 조건 → 항상 검증
- 표준 근사(SIE)는 프로젝트 전역에서 단일 고정 — 함수에 `approximation_*` 인자 추가 금지
- 모든 물리 함수 → 단위 + 표준 근사 가정 명시 docstring 필수
  (예: "SIE 표준 근사 가정", "full_numerical 모드 — truth 생성용")

**검증 기준 (변경 불가)**
| 벤치마크 | 통과 기준 |
|---------|---------|
| system 6 | Δt 오차 < 0.15일 |
| ZTF 노이즈 | precision ≥ 92.3%, recall ≥ 60% |
| SDSS J1226-0006 | COSMOGRAIL 1σ 이내 |
| TDC1 전체 | 오차 < 3% |

---

## 세션 종료 — 문서 업데이트

작업 후 반드시 출력:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 문서 업데이트 

[CHANGELOG.md] ## [날짜] 요약
[STATUS.md] 변경 항목
[ARCHITECTURE.md] 구조 변경 시만
[DECISIONS.md] 선택 결정 시만
[BENCHMARKS.md] 검증 실행 시만

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

규칙:
- 작업 완료 후에만 출력 (작업 중 끊지 말 것)
