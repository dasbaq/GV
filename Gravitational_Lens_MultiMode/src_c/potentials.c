#include "core_engine.h"
#include <math.h>

// 1. 기존 중심 은하 (SIE 모델)의 굴절률
double get_elliptical_refractive_index(Vec3 pos, SimConfig config) {
  (void)config;
  double dx = pos.x;
  double dy = pos.y;
  double q = 0.7;
  double core = 0.1;
  double s = sqrt(q * q * dx * dx + dy * dy + core * core);
  return 1.0 + (0.5 / s);
}

// 2. ✨ [핵심] 모드별 다중 중력원 통합 계산 함수
double get_total_refractive_index(Vec3 pos, SimConfig config) {
  double n_base = get_elliptical_refractive_index(pos, config);
  double n_sub = 0.0;

  if (config.mode == 2) {
    // Mode 2: 암흑물질 (NFW 유사 프로파일)
    // config.yaml의 sub_pos [2.0, 1.5] 위치에 존재한다고 가정
    double sub_x = 2.0;
    double sub_y = 1.5;
    double r = sqrt(pow(pos.x - sub_x, 2) + pow(pos.y - sub_y, 2) + 0.1);
    // 암흑물질은 넓게 퍼져 빛을 완만하게 꺾음
    n_sub = 0.05 / (r * (1.0 + r));
  } else if (config.mode == 3) {
    // Mode 3: 초거대 블랙홀 (Point Mass)
    // 중심(0, 0)에 위치한다고 가정
    double r = sqrt(pos.x * pos.x + pos.y * pos.y + 1e-4);
    // 블랙홀은 매우 좁은 영역에서 빛을 극단적으로 꺾음 ($1/r$ 형태)
    n_sub = 0.2 / r;
  }

  return n_base + n_sub;
}

// 3. 굴절률의 기울기(Gradient) 계산 -> 빛을 꺾는 힘
Vec3 get_refractive_index_gradient(Vec3 pos, SimConfig config) {
  double eps = 1e-5;
  Vec3 grad;

  Vec3 px = {pos.x + eps, pos.y, pos.z};
  Vec3 mx = {pos.x - eps, pos.y, pos.z};
  // 기존 get_elliptical_refractive_index 대신 get_total_refractive_index 사용
  grad.x = (get_total_refractive_index(px, config) -
            get_total_refractive_index(mx, config)) /
           (2.0 * eps);

  Vec3 py = {pos.x, pos.y + eps, pos.z};
  Vec3 my = {pos.x, pos.y - eps, pos.z};
  grad.y = (get_total_refractive_index(py, config) -
            get_total_refractive_index(my, config)) /
           (2.0 * eps);

  grad.z = 0.0;
  return grad;
}