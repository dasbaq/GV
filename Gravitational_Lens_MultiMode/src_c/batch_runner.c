// src_c/batch_runner.c

#include "core_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ✅ int mode 대신 SimConfig config를 받도록 수정 (core_engine.h와 통일)
void run_batch_simulation(SimConfig config, const char *filename) {
  // 이제 에러 없이 config에서 mode를 꺼낼 수 있습니다.
  int mode = config.mode;

  FILE *file = fopen(filename, "w");
  if (file == NULL)
    return;

  // 정답지인 true_H0를 함께 기록
  fprintf(file, "init_x,init_y,final_x,final_y,time_delay,true_H0\n");
  srand((unsigned int)time(NULL));

  for (int i = 0; i < 10000; i++) {
    double init_x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    double init_y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    double final_x = 0.0, final_y = 0.0, base_t_delay = 0.0;

    // config를 첫 번째 인자로 전달!
    trace_ray(config, init_x, init_y, &final_x, &final_y, &base_t_delay);

    fprintf(file, "%f,%f,%f,%f,%f,%f\n", init_x, init_y, final_x, final_y,
            base_t_delay, TRUE_H0);
  }

  // 완료 메시지 스케일업
  printf("✅ [Batch] 10000개의 데이터 생성 완료. (Mode: %d)\n", mode);
}