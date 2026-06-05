import subprocess
import sys
from pathlib import Path
import shutil

def main():
    print("=" * 80)
    print(" VMEM BENCHMARK - END-TO-END INTEGRATION TEST")
    print("=" * 80)
    
    # Define test directories
    test_out_dir = Path("d:/Perdue/test_outputs")
    if test_out_dir.exists():
        print(f"Cleaning existing test directory {test_out_dir}...")
        try:
            shutil.rmtree(test_out_dir)
        except Exception as e:
            print(f"Warning: could not clean test directory completely: {e}")
        
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Starting feature extraction (extract.py) for test...")
    # Run extract.py on 1 sequence, with 2 corruptions at severities 1 and 5
    extract_cmd = [
        sys.executable,
        "vmem_benchmark/extract.py",
        "--max-seq", "1",
        "--corruptions", "hot_pixel", "event_flood",
        "--severities", "1", "5",
        "--output-dir", str(test_out_dir)
    ]
    print(f"Running command: {' '.join(extract_cmd)}")
    
    res = subprocess.run(extract_cmd, capture_output=True, text=True, cwd="d:/Perdue")
    print("\n--- EXTRACTION STDOUT ---")
    print(res.stdout)
    print("--- EXTRACTION STDERR ---")
    print(res.stderr)
    
    if res.returncode != 0:
        print("Extraction failed!")
        sys.exit(1)
        
    print("\n[2] Starting downstream analysis (analyse.py) for test...")
    # Run analyse.py on the extracted features
    analyse_cmd = [
        sys.executable,
        "vmem_benchmark/analyse.py",
        "--output-dir", str(test_out_dir),
        "--fast"
    ]
    print(f"Running command: {' '.join(analyse_cmd)}")
    
    res_analyse = subprocess.run(analyse_cmd, capture_output=True, text=True, cwd="d:/Perdue")
    print("\n--- ANALYSIS STDOUT ---")
    print(res_analyse.stdout)
    print("--- ANALYSIS STDERR ---")
    print(res_analyse.stderr)
    
    if res_analyse.returncode != 0:
        print("Analysis failed!")
        sys.exit(1)
        
    print("\n[3] Checking outputs...")
    expected_files = [
        test_out_dir / "plots" / "pca_subspaces.pdf",
        test_out_dir / "tables" / "severity_regression.csv",
        test_out_dir / "plots" / "corruption_confusion_matrix.pdf",
        test_out_dir / "tables" / "corruption_classification_report.csv"
    ]
    
    missing = []
    for f in expected_files:
        if f.exists():
            print(f"  [FOUND] {f.name} ({f.stat().st_size} bytes)")
        else:
            print(f"  [MISSING] {f.name}")
            missing.append(f)
            
    if missing:
        print("\nTest failed: Some expected files are missing!")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("SUCCESS: Full top-to-bottom extract-to-analyse test passed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
