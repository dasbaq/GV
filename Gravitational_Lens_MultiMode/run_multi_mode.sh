#!/bin/bash
MODE=$1
if [ -z "$MODE" ]; then 
    echo "오류: 실행 모드 번호를 입력해주세요."
    exit 1
fi

# ✨ 해결의 핵심: 결과를 저장할 폴더를 미리 만들어 둡니다!
mkdir -p data/outputs

echo "--- Step 1: Synchronizing Configurations ---"
cd src_py || exit
python3 sync_config.py
cd ..

echo "--- Step 2: Rebuilding C-Engine ---"
cd src_c || exit
make clean && make all
cd ..
# ✨ 해결의 핵심 2: C엔진을 프로젝트 최상단(루트)에서 실행하여 경로를 맞춥니다.
./src_c/bin/raytrace_engine $MODE

echo "--- Step 3: Launching Python Analytics ---"
cd src_py || exit
python3 main.py --mode $MODE
cd ..

echo "--- Pipeline Execution Completed! ---"
echo "--- Step 4: Running ML Training Pipeline ---"
python3 src_py/ml/ml_pipeline.py $MODE

echo "--- All Pipeline Steps Completed! ---"