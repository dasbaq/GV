# src_py/test_lightcurve.py
import numpy as np
import matplotlib.pyplot as plt
from core.lightcurve import LightCurveSimulator

def main():
    print("🌌 퀘이사 DRW 광도곡선 및 렌즈 합성 시뮬레이션 시작...")
    
    # 1. 관측 스케줄 세팅 (3년간, 3일에 한 번씩 관측한다고 가정)
    t_obs = np.arange(0, 1000, 3.0)
    
    # 2. 시뮬레이터 초기화 및 원본 퀘이사 생성
    simulator = LightCurveSimulator(tau=150.0, sigma=0.5, mean_flux=20.0)
    source_flux = simulator.generate_drw_source(t_obs)
    
    # 3. 렌즈 효과 적용 (가상의 C 엔진 출력값 적용)
    # 상 A는 빨리 도착하고(t=0), 상 B는 45일 뒤에 도착함 (t=45)
    # 상 A가 상 B보다 1.5배 더 밝음
    t_A, t_B = 0.0, 45.0
    mu_A, mu_B = 1.5, 1.0 
    
    joint_flux, flux_A, flux_B = simulator.generate_unresolved_curve(
        t_obs, source_flux, t_A, t_B, mu_A, mu_B
    )
    
    # 4. 시각화
    plt.figure(figsize=(12, 6))
    
    # 분해 가능한 이상적인 경우 (점선)
    plt.plot(t_obs, flux_A, '--', label='Image A (mu=1.5, t=0)', color='cyan', alpha=0.7)
    plt.plot(t_obs, flux_B, '--', label='Image B (mu=1.0, t=45)', color='magenta', alpha=0.7)
    
    # 우리가 실제로 관측하게 될 미분해 데이터 (실선)
    plt.plot(t_obs, joint_flux, '-', label='Unresolved Joint Flux (What we see)', color='yellow', linewidth=2)
    
    plt.title('Simulated Unresolved Quasar Light Curve (DRW Model)')
    plt.xlabel('Observation Time [Days]')
    plt.ylabel('Flux [Arbitrary Units]')
    plt.style.use('dark_background')
    plt.legend()
    plt.grid(alpha=0.2)
    
    # 이미지 저장
    plt.savefig('../data/outputs/drw_lightcurve_test.png')
    print("✅ 완료! '../data/outputs/drw_lightcurve_test.png' 파일을 확인해보세요.")

if __name__ == "__main__":
    main()