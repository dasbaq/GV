# src_py/main.py (Patch Update)
# Missing Component 2: Routing the subclasses and Visualizer

import argparse
from core.data_loader import DataLoader
from modes.cosmology import CosmologyMode
from modes.darkmatter import DarkMatterMode
from modes.smbh import SMBHMode
from ml.regressor import TargetAgnosticRegressor
from visualization import Visualizer  # Added Import

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, required=True)
    args = parser.parse_args()
    
    csv_path = f"../data/outputs/raytrace_mode_{args.mode}.csv"
    loader = DataLoader(csv_path)
    df = loader.load_trajectory_data()
    
    # 1. Visualization Subroutine
    print("[Python Pipeline] Generating 3D Projections...")
    vis = Visualizer(df)
    vis.plot_time_delay_surface(output_path=f"../data/outputs/surface_mode_{args.mode}.png")
    
    # 2. Polymorphic Routing
    if args.mode == 1:
        simulator = CosmologyMode()
    elif args.mode == 2:
        simulator = DarkMatterMode()
    elif args.mode == 3:
        simulator = SMBHMode()
    else:
        raise ValueError("Invalid Mode ID.")
        
    simulator.load_data(df)
    X, Y = simulator.preprocess_features()
    
    # 3. ML Pipeline
    regressor = TargetAgnosticRegressor()
    regressor.train_and_evaluate(X, Y)

if __name__ == "__main__":
    main()