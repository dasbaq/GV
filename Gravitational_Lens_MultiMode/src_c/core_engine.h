#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H

// 1. 3차원 벡터 구조체
typedef struct {
  double x;
  double y;
  double z;
} Vec3;

// 2. 빛(Ray) 구조체
typedef struct {
  Vec3 pos; // 현재 위치
  Vec3 vel; // 현재 진행 방향 (속도)
  double time_delay;
} Ray;

// 3. 시뮬레이션 설정 구조체 (반드시 params.h보다 먼저 선언되어야 함!)
typedef struct {
  int mode;
} SimConfig;

// 4. 파이썬이 자동 생성하는 물리 파라미터 헤더 불러오기
#include "params.h"

// 5. 물리 및 광선 추적 함수 선언
double get_total_refractive_index(Vec3 pos, SimConfig config);
Vec3 get_refractive_index_gradient(Vec3 pos, SimConfig config);
void step_ray_rk4(Ray *ray, SimConfig config, double dt);
// 진짜 빛을 추적하는 메인 엔진 함수 선언 추가
void trace_ray(SimConfig config, double init_x, double init_y, double *final_x,
               double *final_y, double *base_t_delay);
// 6. 배치 실행 함수 선언
void run_batch_simulation(SimConfig config, const char *output_filepath);

#endif