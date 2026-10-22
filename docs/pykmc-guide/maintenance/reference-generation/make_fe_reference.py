"""Rebuild the supplied Fe four-vacancy example's 6750-site analysis reference."""
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
    if (len(atoms) != 6746 or not np.all(atoms.numbers == 26)
            or not np.isfinite(atoms.positions).all() or not atoms.pbc.all()
            or not np.allclose(atoms.cell.array, np.eye(3) * 42.8295, atol=1e-6, rtol=0)):
        raise ValueError("Expected 6746 Fe atoms in a periodic 42.8295 A cube")
    reference = bulk("Fe", "bcc", a=2.8553, cubic=True).repeat((15, 15, 15))
    box = reference.cell.lengths()
    tree = cKDTree(reference.positions % box, boxsize=box)
    distance, sites = tree.query(atoms.positions % box)
    occupancy = np.bincount(sites, minlength=6750)
    if ((occupancy == 0).sum() != 4 or np.maximum(occupancy - 1, 0).sum() != 0
            or distance.max() > 0.5):
        raise ValueError("Input does not match the aligned four-vacancy Fe example")
    with args.output.open("x") as handle:
        write(handle, reference, format="extxyz")
    print("Wrote 6750 ideal BCC Fe sites; analysis reference, not a simulation input.")


if __name__ == "__main__":
    main()
