# src_py/sync_config.py
# Phase 7: Automated Configuration Synchronizer

import yaml
import os

def generate_c_header(config_path: str, header_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")

    # Load YAML Configuration
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)

    # Generate C Header File
    with open(header_path, 'w') as f:
        f.write("/* AUTO-GENERATED CONFIGURATION HEADER */\n")
        f.write("/* DO NOT EDIT DIRECTLY. MODIFY config.yaml INSTEAD. */\n")
        f.write("#ifndef PARAMS_H\n#define PARAMS_H\n\n")
        
        # Physics Constants
        f.write(f"#define C_LIGHT {conf['physics']['c_light']}\n")
        f.write(f"#define EPSILON {conf['physics']['epsilon']}\n")
        
        # Simulation Parameters
        f.write(f"#define STEP_SIZE {conf['simulation']['step_size']}\n")
        f.write(f"#define MAX_STEPS {conf['simulation']['max_steps']}\n")
        f.write(f"#define GRID_RES {conf['simulation']['grid_resolution']}\n")
        f.write(f"#define SOURCE_Z {conf['simulation']['source_z']}\n")
        
        # Lens Model Parameters
        f.write(f"#define MAIN_MASS {conf['lens_model']['main_mass']}\n")
        #Cosmology Parameters 파싱 및 C-Header 변환
        f.write("\n// Cosmology Parameters\n")
        f.write(f"#define TRUE_H0 {conf['cosmology']['true_H0']}\n")
        f.write(f"#define Z_LENS {conf['cosmology']['z_lens']}\n")
        f.write(f"#define Z_SOURCE {conf['cosmology']['z_source']}\n")
        f.write(f"#define OMEGA_M {conf['cosmology']['omega_m']}\n")
        f.write(f"#define OMEGA_LAMBDA {conf['cosmology']['omega_lambda']}\n")
        
        f.write("\n#endif // PARAMS_H\n")
        
    print(f"[Sync] Successfully generated C-Header: {header_path}")

if __name__ == "__main__":
    # Execute relative to the src_py directory
    generate_c_header('config.yaml', '../src_c/params.h')