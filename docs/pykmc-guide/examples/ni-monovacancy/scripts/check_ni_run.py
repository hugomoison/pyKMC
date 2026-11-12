"""Read-only consistency checks for a fresh, fixed-cell Ni monovacancy run.

Requires NumPy and ASE; optional site-occupancy checks also require SciPy.
Pair displacements assume unchanged atom ordering across trajectory frames.
These checks do not establish event-catalog completeness or physical accuracy.
"""
import argparse
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, read


def check_progress(path, steps):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            rows.append(fields)
    if [int(row[0]) for row in rows] != list(range(steps + 1)):
        raise ValueError("Progress steps are missing, duplicated, or do not end at the requested step.")
    times = []
    for row in rows:
        values = np.asarray(row[1:], dtype=float)
        if not np.isfinite(values).all() or len(values) < 2:
            raise ValueError("Invalid progress values.")
        times.append(values[1])
        if int(row[0]) > 0:
            if len(row) != 10 or values[0] <= 0 or values[4] <= 0 or values[5] <= 0:
                raise ValueError("Unexpected progress format or nonpositive time/rates.")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Cumulative KMC time does not increase from zero.")
    print(f"PASS: progress steps 0..{steps}, finite values and increasing KMC time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--pair", nargs=2, type=int, action="append", metavar=("START", "END"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    pairs = args.pair or [(0, 1)]
    if any(a < 0 or b != a + 1 or b > args.steps for a, b in pairs):
        parser.error("Each pair must be consecutive frames within 0..steps")
    wanted = {i for pair in pairs for i in pair}
    frames = {}
    vacancy_positions = {}
    expected_cell = np.eye(3) * 28.16
    tree = None
    if args.reference:
        from scipy.spatial import cKDTree
        reference = read(args.reference)
        if (len(reference) != 2048 or not np.all(reference.numbers == 28)
                or not reference.pbc.all() or not np.isfinite(reference.positions).all()
                or not np.allclose(reference.cell.array, expected_cell, atol=1e-6, rtol=0)):
            raise ValueError("Expected a finite 2048-site periodic Ni reference with the same cell.")
        box = reference.cell.lengths()
        tree = cKDTree(reference.positions % box, boxsize=box)
        if tree.query(reference.positions % box, k=2)[0][:, 1].min() < 1e-5:
            raise ValueError("Duplicate reference sites.")

    count = 0
    for index, atoms in enumerate(iread(args.trajectory, index=":")):
        if (len(atoms) != 2047 or not np.all(atoms.numbers == 28)
                or not np.isfinite(atoms.positions).all() or not atoms.pbc.all()
                or not np.allclose(atoms.cell.array, expected_cell, atol=1e-6, rtol=0)):
            raise ValueError(f"Frame {index}: atom/species/coordinate/cell/PBC check failed")
        if tree is not None:
            _, sites = tree.query(atoms.positions % box)
            occupancy = np.bincount(sites, minlength=2048)
            vacancies = np.count_nonzero(occupancy == 0)
            interstitials = np.maximum(occupancy - 1, 0).sum()
            if vacancies != 1 or interstitials != 0:
                raise ValueError(f"Frame {index}: {vacancies} vacancies, {interstitials} excess atoms; review reference and run")
            if index in wanted:
                site_index = int(np.flatnonzero(occupancy == 0)[0])
                vacancy_positions[index] = reference.positions[site_index] % box
        if index in wanted:
            frames[index] = atoms.copy()
        count += 1
    if count != args.steps + 1:
        raise ValueError(f"Expected {args.steps+1} frames, found {count}")
    print(f"PASS: {count} readable frames; each has 2047 Ni atoms, finite positions, fixed periodic 28.16 A cell")
    if tree is not None:
        print("PASS: nearest-reference-site counts: one vacancy, zero excess atoms in every frame")
    if args.progress:
        check_progress(args.progress, args.steps)

    print("Displacement results below assume consistent atom ordering (indices are zero-based).")
    results = {}
    for start, end in pairs:
        before, after = frames[start], frames[end]
        if tree is not None:
            raw = vacancy_positions[end] - vacancy_positions[start]
            vacancy_vector, vacancy_length = find_mic(raw, before.cell, before.pbc)
            print(f"\nVacancy sites, frame {start} -> {end} (A):",
                  np.round(vacancy_positions[start], 6), '->',
                  np.round(vacancy_positions[end], 6))
            print("Wrapped-coordinate difference(A):", np.round(raw, 6))
            print("Minimum-image vacancy vector(A):", np.round(vacancy_vector, 6),
                  "length(A):", f"{float(vacancy_length):.6f}")
        vectors, lengths = find_mic(after.positions - before.positions, before.cell, before.pbc)
        order = np.argsort(lengths)[::-1][:3]
        print(f"\nFrame {start} -> {end}: index, displacement(A), vector(A)")
        for index in order:
            print(int(index), f"{lengths[index]:.6f}", np.round(vectors[index], 6))
        print("Atoms moving more than 1 A:", np.count_nonzero(lengths > 1))
        results[(start, end)] = (vectors, lengths, int(order[0]))
    for (start, middle), (vectors1, lengths1, atom1) in results.items():
        if (middle, middle + 1) not in results:
            continue
        vectors2, lengths2, atom2 = results[(middle, middle + 1)]
        net = vectors1 + vectors2
        print(f"\nTwo-step check {start} -> {middle} -> {middle+1}")
        print("Same largest-moving atom:", atom1 == atom2)
        print("Maximum two-step net displacement(A):", f"{np.linalg.norm(net, axis=1).max():.6f}")
        if atom1 == atom2 and lengths1[atom1] * lengths2[atom1] > 0:
            cosine = np.dot(vectors1[atom1], vectors2[atom1]) / (lengths1[atom1] * lengths2[atom1])
            print("Jump-vector cosine:", f"{np.clip(cosine, -1, 1):.6f}")


if __name__ == "__main__":
    main()
