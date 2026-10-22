"""Rebuild the analysis asset for the supplied 511-atom Si vacancy cell only."""
import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import read, write
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    atoms = read(args.initial)
    if (len(atoms) != 511 or not np.all(atoms.numbers == 14)
            or not atoms.pbc.all() or not np.isfinite(atoms.positions).all()
            or not np.allclose(atoms.cell.array, np.eye(3) * 21.72, atol=1e-6, rtol=0)):
        raise ValueError("Expected the 511-atom Si vacancy input in its aligned periodic 21.72 A cube")
    reference = bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))
    box = reference.cell.lengths()
    tree = cKDTree(reference.positions % box, boxsize=box)
    distances, sites = tree.query(atoms.positions % box)
    occupancy = np.bincount(sites, minlength=512)
    # Guard against a wrong origin/orientation; allow the supplied local relaxation.
    if distances.max() > 0.75 or (occupancy == 0).sum() != 1 or np.any(occupancy > 1):
        raise ValueError("Input is not aligned to this diamond reference; review it before generating an asset")
    with args.output.open("x") as handle:
        write(handle, reference, format="extxyz")
    print("Wrote 512 ideal Si sites in a periodic 21.72 A cubic cell")
    print("This is an analysis reference, not a replacement simulation input.")


if __name__ == "__main__":
    main()
