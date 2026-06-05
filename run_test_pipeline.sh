#!/bin/bash
# run_test_pipeline.sh
# Triggers the python integration runner to test all operations end-to-end.

set -e

echo "Starting full integration pipeline test..."
./vmem_benchmark/.venv/Scripts/python.exe run_test_pipeline.py
