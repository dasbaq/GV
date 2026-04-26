# src_py/visualization.py
# Phase 4: Multi-mode 3D plotting & Time Delay curve utility

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ✅ Fix 1: output_path에 'str' 대신 'Optional[str]' 사용
    def plot_3d_trajectories(self, output_path: Optional[str] = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for rendering performance
        sample_size = min(500, len(self.df))
        plot_df = self.df.sample(n=sample_size, random_state=42)

        # Iterate via DataFrame rows for trajectory projection
        for _, row in plot_df.iterrows():
            # ✅ Fix 2: 잠재적인 KeyError 방지를 위해 C 엔진 출력(소문자)과 맞춤
            # (만약 C 엔진에 final_z가 없다면 여기서 에러가 날 수 있으니 주의하세요!)
            xs = [row.get('init_x', 0), row.get('final_x', 0)]
            ys = [row.get('init_y', 0), row.get('final_y', 0)]
            zs = [-50.0, row.get('final_z', 0)] # Z-axis boundary assumption
            
            ax.plot(xs, ys, zs, alpha=0.3, color='royalblue')

        ax.set_xlabel('X_Axis')
        ax.set_ylabel('Y_Axis')
        # ✅ Fix 3: 3D 전용 메서드에 대한 Pylance 경고 무시 처리
        ax.set_zlabel('Z_Axis')  # type: ignore
        ax.set_title('3D Ray Trajectories (Linear Interpolation)')

        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()

    # ✅ Fix 4: output_path에 'str' 대신 'Optional[str]' 사용
    def plot_time_delay_surface(self, output_path: Optional[str] = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = self.df['final_x']
        y = self.df['final_y']
        z = self.df['time_delay']

        # Triangular surface plot for irregular data grids
        # ✅ Fix 5: 3D 전용 메서드에 대한 Pylance 경고 무시 처리
        surf = ax.plot_trisurf(x, y, z, cmap='inferno', edgecolor='none')  # type: ignore
        fig.colorbar(surf, ax=ax, label='Accumulated Time Delay')

        ax.set_xlabel('Observer X')
        ax.set_ylabel('Observer Y')
        # ✅ Fix 6: 3D 전용 메서드에 대한 Pylance 경고 무시 처리
        ax.set_zlabel('Time Delay [Arbitrary Units]')  # type: ignore
        ax.set_title('Time Delay Surface Map')

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()