"""Read-only checks for a fresh Si monovacancy trajectory in the supplied cell.

Requires NumPy, ASE and SciPy. This is an output-consistency check, not proof
of a complete event catalogue or accurate material kinetics.
"""
import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import iread, read
from scipy.spatial import cKDTree


CELL_LENGTH = 21.72
CELL = np.eye(3) * CELL_LENGTH


def check_structure(atoms, count, label):
    if (len(atoms) != count or not np.all(atoms.numbers == 14)
            or not np.isfinite(atoms.positions).all() or not atoms.pbc.all()
            or not np.allclose(atoms.cell.array, CELL, atol=1e-6, rtol=0)):
        raise ValueError(f"{label}: expected {count} Si atoms, finite positions and a periodic 21.72 A cube")


def check_progress(path, steps):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            rows.append(fields)
    if [int(row[0]) for row in rows] != list(range(steps + 1)):
        raise ValueError("Progress steps must be exactly 0 through the requested final step")
    if len(rows[0]) != 3 or not np.array_equal(np.asarray(rows[0][1:], dtype=float), [0, 0]):
        raise ValueError("Expected a fresh step-0 row with zero physical time")
    previous_time = 0.0
    for row in rows[1:]:
        if len(row) != 10:
            raise ValueError(f"Step {row[0]}: expected ten progress columns")
        values = np.asarray(row, dtype=float)
        _, dt, total, ref_id, barrier, event_rate, total_rate, _, cpu, wall = values
        if (not np.isfinite(values).all() or dt <= 0 or total <= previous_time
                or ref_id < 0 or ref_id != int(ref_id) or barrier < 0
                or event_rate <= 0 or total_rate <= 0 or cpu < 0 or wall < 0
                or event_rate > total_rate * (1 + 1e-5)):
            raise ValueError(f"Step {row[0]}: invalid progress values, time or rates")
        # Text columns have limited precision; allow rounding in accumulated time.
        tolerance = 3e-6 * max(abs(total), abs(previous_time), dt)
        if abs(total - previous_time - dt) > tolerance:
            raise ValueError(f"Step {row[0]}: dT does not match the cumulative-time increment")
        previous_time = total
    print(f"PASS: progress steps 0..{steps} with positive rates and increasing KMC time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    try:
        reference = read(args.reference)
        check_structure(reference, 512, "Reference")
        box = np.full(3, CELL_LENGTH)
        tree = cKDTree(reference.positions % box, boxsize=box)
        ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))
        distances, indices = tree.query(ideal.positions % box)
        if distances.max() > 1e-6 or len(np.unique(indices)) != 512:
            raise ValueError("Reference is not the supplied aligned 512-site diamond lattice")
        frame_count = 0
        vacancy_positions = []
        for index, atoms in enumerate(iread(args.trajectory, index=":")):
            check_structure(atoms, 511, f"Frame {index}")
            _, indices = tree.query(atoms.positions % box)
            occupancy = np.bincount(indices, minlength=512)
            vacancies = np.flatnonzero(occupancy == 0)
            excess = int(np.maximum(occupancy - 1, 0).sum())
            if len(vacancies) != 1 or excess:
                raise ValueError(f"Frame {index}: {len(vacancies)} vacant sites and {excess} excess atoms; review the reference and run")
            vacancy_positions.append(reference.positions[vacancies[0]] % box)
            frame_count += 1
        if frame_count != args.steps + 1:
            raise ValueError(f"Expected {args.steps + 1} frames, found {frame_count}")
        print(f"PASS: {frame_count} frames; 511 Si atoms and a fixed periodic 21.72 A cell")
        print("PASS: one vacant reference site and zero excess atoms in every frame")
        check_progress(args.progress, args.steps)
        hops = np.diff(np.asarray(vacancy_positions), axis=0)
        hops -= box * np.rint(hops / box)
        lengths = np.linalg.norm(hops, axis=1)
        print(f"INFO: minimum-image vacancy-site step lengths: {lengths.min():.6f} .. {lengths.max():.6f} A")
        print("Check logs and event rates as well; these checks do not establish kinetic accuracy.")
    except (OSError, ValueError, IndexError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
