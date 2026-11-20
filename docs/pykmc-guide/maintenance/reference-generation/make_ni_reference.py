"""Create a WS reference for the tutorial's aligned 8x8x8 FCC Ni cell only."""
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
    if len(atoms) != 2047 or not np.all(atoms.numbers == 28):
        raise ValueError("This helper requires the 2047-atom Ni monovacancy example.")
    if not atoms.pbc.all() or not np.allclose(
        atoms.cell.array, np.eye(3) * 28.16, atol=1e-6, rtol=0
    ):
        raise ValueError("Expected the aligned periodic 28.16 A cubic cell.")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Non-finite input coordinates.")

    reference = bulk("Ni", "fcc", a=3.52, cubic=True).repeat((8, 8, 8))
    box = reference.cell.lengths()
    tree = cKDTree(reference.positions % box, boxsize=box)
    distances, sites = tree.query(atoms.positions % box)
    occupancy = np.bincount(sites, minlength=len(reference))
    # Do not silently generate a misaligned reference for a different structure.
    if distances.max() > 0.15 or np.count_nonzero(occupancy == 0) != 1 or np.any(occupancy > 1):
        raise ValueError("Input is not aligned to this FCC reference; review origin/orientation/structure.")
    # Exclusive creation protects the input and any existing reference file.
    with args.output.open("x") as handle:
        write(handle, reference, format="extxyz")
    print(f"Wrote {args.output}: 2048 ideal Ni sites, periodic 28.16 A cubic cell")
    print(f"Input mapping: one empty site, no multiply occupied sites; max offset {distances.max():.6f} A")
    print("This is a visualization reference, NOT a replacement simulation input.")


if __name__ == "__main__":
    main()
