"""
Backwards-compatible entry point. All analysis logic has moved to analysis/analyse.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path so analysis module can be imported properly
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.analyse import main

if __name__ == "__main__":
    main()
