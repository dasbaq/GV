#include "core_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // 1. 실행 모드(Mode) 전달 확인
  if (argc < 2) {
    printf("Error: Missing mode argument.\n");
    return 1;
  }

  // 2. 현재 최신화된 파이프라인에 맞춘 시뮬레이션 설정
  SimConfig config;
  config.mode = atoi(argv[1]);

  printf("Engine Initialization Complete. Mode: %d\n", config.mode);
  printf("Invoking Batch Runner...\n");

  // 3. 파이썬이 읽어갈 수 있도록 출력 파일 경로 지정
  char output_filepath[256];
  sprintf(output_filepath, "data/outputs/raytrace_mode_%d.csv", config.mode);

  // 4. 빛의 궤적 계산 시뮬레이션(batch_runner.c) 가동
  run_batch_simulation(config, output_filepath);

  return 0;
}