import re

def main():
    log_path = "d:/Perdue/test_pipeline_live.log"
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines, 1):
        # Filter out progress bar lines to make it readable
        if '%' in line and ('|' in line or 'seq/s' in line or 'it/s' in line):
            continue
        # Also clean up multiple empty lines
        line_stripped = line.strip()
        if not line_stripped:
            continue
        print(f"{i}: {line_stripped}")

if __name__ == "__main__":
    main()
