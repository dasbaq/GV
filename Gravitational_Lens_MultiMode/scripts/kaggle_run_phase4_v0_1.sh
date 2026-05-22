#!/usr/bin/env bash
# Kaggle notebook 셀에서 실행할 Phase 4 v0.1 train 명령.
# 사용법(Kaggle notebook):
#   !bash /kaggle/working/repo/Gravitational_Lens_MultiMode/scripts/kaggle_run_phase4_v0_1.sh
#
# Kaggle Dataset slug가 lens-phase4-v0-1이 아니면 KAGGLE_SLUG를 덮어쓰세요:
#   !KAGGLE_SLUG=my-slug bash .../kaggle_run_phase4_v0_1.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/kaggle/working/repo/Gravitational_Lens_MultiMode}"
KAGGLE_SLUG="${KAGGLE_SLUG:-datasets/donghyun51/lens-phase4-v0-1}"
INPUT_DIR="/kaggle/input/${KAGGLE_SLUG}"

EQUIV_JSON="${INPUT_DIR}/phase4_v0_1_equivalence.json"
UNFILTERED_H5="${INPUT_DIR}/phase4_v0_1_eval_unfiltered.h5"

if [[ ! -f "$UNFILTERED_H5" ]]; then
  echo "[ERR] not found: $UNFILTERED_H5" >&2
  echo "      check: ls ${INPUT_DIR}" >&2
  exit 1
fi

EQUIV_ARG=()
if [[ -f "$EQUIV_JSON" ]]; then
  EQUIV_ARG=(--equivalence-from="$EQUIV_JSON")
else
  echo "[WARN] equivalence JSON missing, proceeding with handoff=not_provided" >&2
fi

cd "$REPO_ROOT"
exec python scripts/phase4_v0_1_round.py \
  --phase=train \
  --device=cuda \
  --workers=4 \
  --epochs=50 \
  --bootstrap-n=1000 \
  "${EQUIV_ARG[@]}" \
  --eval-unfiltered="$UNFILTERED_H5"
