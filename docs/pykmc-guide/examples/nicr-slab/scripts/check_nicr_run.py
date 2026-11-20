"""Check the supplied NiCr trajectory without changing it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from basin_checks import main

if __name__ == "__main__":
    main("nicr")
