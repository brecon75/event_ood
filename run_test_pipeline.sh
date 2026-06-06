#!/bin/bash
# run_test_pipeline.sh
# Runs a fast validation test of the full SNN-ANN benchmark pipeline end-to-end (2 sequences) in Bash.

set -e

CONFIG_FILE="vmem_benchmark/benchmark_config.py"
BACKUP_FILE="vmem_benchmark/benchmark_config.py.bak"

echo "======================================================================="
echo "   PREPARING FAST PIPELINE TEST (Capping sequences to 2)"
echo "======================================================================="

# 1. Back up original config
cp "$CONFIG_FILE" "$BACKUP_FILE"

restore_config() {
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$CONFIG_FILE"
        rm "$BACKUP_FILE"
        echo -e "\033[0;32m--> Restored original configuration.\033[0m"
    fi
}

# Ensure config is restored on exit
trap restore_config EXIT

# 2. Modify config temporarily (replace MAX_SEQUENCES = <number> with 2)
# Using sed with cross-platform compatibility:
sed -i.tmp 's/MAX_SEQUENCES[[:space:]]*=[[:space:]]*[0-9]*/MAX_SEQUENCES  = 2/g' "$CONFIG_FILE"
rm -f "${CONFIG_FILE}.tmp"

# 3. Execute full benchmark forwarding any arguments
./run_full_benchmark.sh "$@"
