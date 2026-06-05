import subprocess
import sys
import shutil
from pathlib import Path

def run_cmd(args, log_file, cwd="d:/Perdue"):
    log_file.write(f"\n======================================================================\n")
    log_file.write(f"STEP: {' '.join(args)}\n")
    log_file.write(f"======================================================================\n")
    log_file.flush()
    print(f"Running: {' '.join(args)}")
    
    # Passing the file object to stdout and stderr streams the output live
    res = subprocess.run(args, stdout=log_file, stderr=log_file, text=True, cwd=cwd)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}: {' '.join(args)}")
        print(f"Please inspect d:/Perdue/test_pipeline_live.log for details.")
        
        # Restore config before exit
        config_path = Path("vmem_benchmark/benchmark_config.py")
        backup_config_path = Path("vmem_benchmark/benchmark_config.py.bak")
        if backup_config_path.exists():
            shutil.copy2(backup_config_path, config_path)
            backup_config_path.unlink()
        sys.exit(1)
    return res

def main():
    print("=" * 80)
    print(" VMEM BENCHMARK - FULL PIPELINE INTEGRATION TEST")
    print("=" * 80)
    
    config_path = Path("vmem_benchmark/benchmark_config.py")
    backup_config_path = Path("vmem_benchmark/benchmark_config.py.bak")
    
    # Define test directory and log path
    test_out_dir = Path("d:/Perdue/vmem_benchmark/test_outputs")
    log_path = Path("d:/Perdue/test_pipeline_live.log")
    
    # Initialize clean log file
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== VMEM BENCHMARK INTEGRATION PIPELINE TEST LOG ===\n")
        
    print(f"Live output will be written to: {log_path}")
    
    # Open log file in append mode to pass to subprocesses
    log_file = open(log_path, 'a', encoding='utf-8')
    
    # 1. Clean test outputs
    if test_out_dir.exists():
        print(f"Cleaning existing test directory {test_out_dir}...")
        shutil.rmtree(test_out_dir, ignore_errors=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Back up config
    print("Backing up benchmark_config.py...")
    shutil.copy2(config_path, backup_config_path)
    
    # 3. Override config path in file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if 'OUTPUT_DIR = Path("d:/Perdue/vmem_benchmark/outputs")' in line:
                new_lines.append('OUTPUT_DIR = Path("d:/Perdue/vmem_benchmark/test_outputs")\n')
            elif 'OUTPUT_DIR = Path("d:\\Perdue\\vmem_benchmark\\outputs")' in line:
                new_lines.append('OUTPUT_DIR = Path("d:/Perdue/vmem_benchmark/test_outputs")\n')
            else:
                new_lines.append(line)
                
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        print("Config overridden successfully.")
        
        # Resolve python executable in the venv
        python_exe = str(Path("d:/Perdue/vmem_benchmark/.venv/Scripts/python.exe"))
        
        # Steps to run in order
        steps = [
            # 1. Extract clean and one corruption (hot_pixel) at severity 5
            [python_exe, "vmem_benchmark/extract.py", "--max-seq", "1", "--corruptions", "hot_pixel", "--severities", "5"],
            # 2. Offline features
            [python_exe, "analysis/extract_offline_features.py"],
            # 3. Feature Fusion
            [python_exe, "analysis/fusion_features.py"],
            # 4. Extract ANN Baselines
            [python_exe, "analysis/extract_ann_baselines.py"],
            # 5. Evaluate ANN Baselines
            [python_exe, "analysis/evaluate_ann_baselines.py"],
            # 6. Fit Detectors
            [python_exe, "analysis/fit_detectors.py"],
            # 7. Evaluate Detectors
            [python_exe, "analysis/evaluate_detectors.py"],
            # 8. Representation Ablation
            [python_exe, "analysis/representation_ablation.py"],
            # 9. Severity
            [python_exe, "analysis/severity.py"],
            # 10. Reliability
            [python_exe, "analysis/reliability.py"],
            # 11. Cross corruption
            [python_exe, "analysis/cross_corruption.py"],
            # 12. Level analyses (PCA, Conformal, Classification, Temporal AE, Severity Reg)
            [python_exe, "analysis/analyse.py", "--fast"],
            # 13. Build tables
            [python_exe, "reporting/build_paper_tables.py"],
            # 14. Build figures
            [python_exe, "reporting/build_paper_figures.py"]
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"\n>>> [{i}/{len(steps)}] Running step...")
            run_cmd(step, log_file)
            
        print("\n" + "=" * 80)
        print("SUCCESS: Full top-to-bottom pipeline test passed successfully!")
        print("=" * 80)
        
    finally:
        log_file.close()
        # 4. Restore original config
        print("\nRestoring benchmark_config.py...")
        if backup_config_path.exists():
            shutil.copy2(backup_config_path, config_path)
            backup_config_path.unlink()
            print("Config restored.")

if __name__ == "__main__":
    main()
