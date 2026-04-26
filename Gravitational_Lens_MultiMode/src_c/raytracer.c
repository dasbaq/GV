#include "core_engine.h"
#include <math.h>

// 실제 4차 Runge-Kutta (RK4) 개념을 적용한 빛의 1스텝 전진 함수
void step_ray_rk4(Ray *ray, SimConfig config, double dt) {
  // k1: 현재 위치(t)에서의 기울기와 속도
  Vec3 pos_k1 = ray->pos;
  Vec3 v_k1 = ray->vel;
  Vec3 a_k1 = get_refractive_index_gradient(pos_k1, config);

  // k2: 중간점(t + dt/2)에서의 기울기와 속도 (k1 사용)
  Vec3 pos_k2 = {
      ray->pos.x + v_k1.x * (dt / 2.0),
      ray->pos.y + v_k1.y * (dt / 2.0),
      ray->pos.z + v_k1.z * (dt / 2.0)
  };
  Vec3 v_k2 = {
      ray->vel.x + a_k1.x * (dt / 2.0),
      ray->vel.y + a_k1.y * (dt / 2.0),
      ray->vel.z + a_k1.z * (dt / 2.0)
  };
  Vec3 a_k2 = get_refractive_index_gradient(pos_k2, config);

  // k3: 또 다른 중간점(t + dt/2)에서의 기울기와 속도 (k2 사용)
  Vec3 pos_k3 = {
      ray->pos.x + v_k2.x * (dt / 2.0),
      ray->pos.y + v_k2.y * (dt / 2.0),
      ray->pos.z + v_k2.z * (dt / 2.0)
  };
  Vec3 v_k3 = {
      ray->vel.x + a_k2.x * (dt / 2.0),
      ray->vel.y + a_k2.y * (dt / 2.0),
      ray->vel.z + a_k2.z * (dt / 2.0)
  };
  Vec3 a_k3 = get_refractive_index_gradient(pos_k3, config);

  // k4: 끝점(t + dt)에서의 기울기와 속도 (k3 사용)
  Vec3 pos_k4 = {
      ray->pos.x + v_k3.x * dt,
      ray->pos.y + v_k3.y * dt,
      ray->pos.z + v_k3.z * dt
  };
  Vec3 v_k4 = {
      ray->vel.x + a_k3.x * dt,
      ray->vel.y + a_k3.y * dt,
      ray->vel.z + a_k3.z * dt
  };
  Vec3 a_k4 = get_refractive_index_gradient(pos_k4, config);

  // 최종 위치 및 속도 업데이트 (가중 평균)
  ray->pos.x += (dt / 6.0) * (v_k1.x + 2.0 * v_k2.x + 2.0 * v_k3.x + v_k4.x);
  ray->pos.y += (dt / 6.0) * (v_k1.y + 2.0 * v_k2.y + 2.0 * v_k3.y + v_k4.y);
  ray->pos.z += (dt / 6.0) * (v_k1.z + 2.0 * v_k2.z + 2.0 * v_k3.z + v_k4.z);

  ray->vel.x += (dt / 6.0) * (a_k1.x + 2.0 * a_k2.x + 2.0 * a_k3.x + a_k4.x);
  ray->vel.y += (dt / 6.0) * (a_k1.y + 2.0 * a_k2.y + 2.0 * a_k3.y + a_k4.y);
  ray->vel.z += (dt / 6.0) * (a_k1.z + 2.0 * a_k2.z + 2.0 * a_k3.z + a_k4.z);
}

// 매개변수로 SimConfig config를 받도록 수정!
void trace_ray(SimConfig config, double init_x, double init_y, double *final_x,
               double *final_y, double *base_t_delay) {
  // SimConfig dummy_config 삭제됨
  Ray ray;
  ray.pos.x = init_x;
  ray.pos.y = init_y;
  ray.pos.z = SOURCE_Z;

  ray.vel.x = 0.0;
  ray.vel.y = 0.0;
  ray.vel.z = C_LIGHT;
  ray.time_delay = 0.0;

  // 3. 지구(Z=0)에 도착할 때까지 한 걸음씩 이동
  for (int step = 0; step < MAX_STEPS; step++) {
    // 1. 현재 위치의 유효 굴절률 구하기 (시공간의 휘어짐 정도)
    double n = get_total_refractive_index(ray.pos, config);

    // 2. 빛을 한 걸음 전진시킴 (속도와 위치 업데이트)
    step_ray_rk4(&ray, config, STEP_SIZE);

    // 3. 실제로 이동한 3D 기하학적 거리 계산 (피타고라스 정리)
    double dist = sqrt(ray.vel.x * ray.vel.x + ray.vel.y * ray.vel.y +
                       ray.vel.z * ray.vel.z) *
                  STEP_SIZE;

    // ✨ 핵심: 거리(dist)와 굴절률(n)을 모두 반영하여 시간 지연 누적!
    // -> n이 커지면(중력장) 시간이 더 걸리고, dist가 길어져도 시간이 더 걸림
    ray.time_delay += (n * dist) / C_LIGHT;

    if (ray.pos.z >= 0.0) {
      break;
    }
  }

  *final_x = ray.pos.x;
  *final_y = ray.pos.y;
  *base_t_delay = ray.time_delay;
}