"""Read-only output diagnostics shared by the Fe and NiCr chapters.

No pickle loading, minimization, MPI launch or file writes. Structural consistency
and recognized log markers do not certify kinetics or a complete event catalogue.
"""
import argparse
import configparser
from pathlib import Path
import shlex

import numpy as np
from ase.build import bulk
from ase.io import iread, read
from scipy.spatial import cKDTree


def sections(path):
    config = configparser.ConfigParser(interpolation=None)
    with path.open() as handle:
        config.read_file(handle)
    return {name.lower(): config[name] for name in config.sections()}


def progress(path, steps):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.split() and line.split()[0].isdigit()]
    if [int(row[0]) for row in rows] != list(range(steps + 1)):
        raise ValueError("Progress must contain exactly steps 0 through the input step budget")
    if len(rows[0]) != 3 or [float(x) for x in rows[0][1:]] != [0.0, 0.0]:
        raise ValueError("Expected a fresh step-0 row with zero physical time")
    previous = 0.0
    data = []
    for row in rows[1:]:
        if len(row) != 10:
            raise ValueError("Expected ten progress columns; review this source version's output format")
        values = np.array(row, dtype=float)
        _, dt, total, ref, barrier, rate, total_rate, _, cpu, wall = values
        if (not np.isfinite(values).all() or dt <= 0 or total < previous
                or ref < 0 or ref != int(ref) or barrier < 0 or rate <= 0
                or total_rate <= 0 or cpu < 0 or wall < 0
                or rate > total_rate * (1 + 1e-5)):
            raise ValueError(f"Invalid progress values at step {row[0]}")
        # Printed cumulative time can stay equal when dt is below its precision.
        if abs(total - previous - dt) > 3e-6 * max(total, previous, dt):
            raise ValueError(f"Inconsistent dT/T at step {row[0]}")
        previous = total
        data.append(values)
    return np.array(data)


def log_summary(directory, data):
    log = (directory / "pykmc.log").read_text()
    if "End of simulation" not in log:
        raise ValueError("No End of simulation marker in pykmc.log")
    status_file = directory / "run-exit-status.txt"
    if status_file.exists() and status_file.read_text().strip() != "0":
        raise ValueError("Launcher reports a nonzero or invalid exit status")
    attempts = log.count("Exploring the Basin")
    exploration = log.count(":=> Basin fails with error")
    exit_failures = log.count("Reconstruction Exit State Basin fails")
    fallback = log.count("back to original event")
    events = directory / "pykmc.events"
    exits = events.read_text().count("Basin Exit Events (Selected=") if events.exists() else None
    control = sections(directory / "input.in")["control"]
    basin_enabled = control.getboolean("basin", fallback=None)
    print(f"INFO: recognized basin-attempt markers={attempts}, exploration failures={exploration}, "
          f"exit-reconstruction failures={exit_failures}, fallbacks={fallback}")
    print(f"INFO: recognized applied basin-exit records={exits if exits is not None else 'unavailable'}")
    if basin_enabled is None:
        print("REVIEW: basin is not explicit in input.in; check the recorded effective setting for this source version.")
    if attempts == 0 and basin_enabled is not False:
        print("REVIEW: no recognized basin-attempt markers with basin enabled or unspecified. "
              "This cannot distinguish no attempts from changed/insufficient logging; inspect the source and logs.")
    if fallback or exploration or exit_failures or "Reconstruction fails" in log:
        print("REVIEW: failures/fallbacks occurred; a structural PASS is not basin or kinetic validation.")
    if attempts and exits == 0:
        print("REVIEW: no applied basin exit recorded. Do not call this successful acceleration.")
    if not events.exists():
        print("REVIEW: pykmc.events is missing; inspect the complete run before interpreting exits.")
    if attempts and exits is not None and attempts != exits + fallback:
        print("REVIEW: basin marker counts do not balance; inspect this version's logs manually.")
    for marker in ("PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD", "RECONSTRUCTION_INVALID_MIN2"):
        if marker in log:
            print(f"REVIEW: {marker}: {log.count(marker)} log occurrences")
    dt = data[:, 1]
    if dt.max() > 1000 * np.median(dt):
        print(f"REVIEW: large time-increment contrast at step {int(data[np.argmax(dt), 0])}; "
              "inspect candidate barriers, rates and refinement diagnostics before using physical time.")
    print("Confirm scheduler/launcher success independently; no rate-law or catalogue-completeness certification is made.")


def verify_nicr_mapping(path):
    """Cross-check actual LAMMPS type labels, masses and EAM assignments.

    Do not infer the engine's mapping from alphabetic order in input.in. Require
    evidence in the log and check every echoed pair_coeff setup, including any
    reinitializations. No LAMMPS instance or pickle file is opened.
    """
    masses, labels = {}, {}
    pair_style = None
    checked = 0
    expected_mass = {"Cr": 51.9961, "Ni": 58.6934}
    for line in path.read_text().splitlines():
        if not line.startswith(("clear", "create_box ", "mass ", "labelmap atom ",
                                "pair_style ", "pair_coeff ")):
            continue
        words = shlex.split(line, comments=True)
        if words[0] in ("clear", "create_box"):
            masses, labels = {}, {}
            pair_style = None
        elif words[0] == "mass":
            if len(words) != 3 or not words[1].isdigit():
                raise ValueError("Unsupported mass record in lammps.log.0; review type mapping manually")
            masses[int(words[1])] = float(words[2])
        elif words[0] == "labelmap":
            entries = words[2:]
            if len(entries) % 2 or not all(x.isdigit() for x in entries[::2]):
                raise ValueError("Unsupported labelmap record in lammps.log.0")
            labels.update({int(entries[i]): entries[i + 1] for i in range(0, len(entries), 2)})
        elif words[0] == "pair_style":
            pair_style = " ".join(words[1:])
        elif words[0] == "pair_coeff":
            if (pair_style != "eam/alloy" or len(words) != 6 or words[1:3] != ["*", "*"]
                    or Path(words[3]).name != "Bonny_2013_NiFeCr.eam"
                    or set(labels) != {1, 2} or set(masses) != {1, 2}
                    or set(labels.values()) != {"Cr", "Ni"}):
                raise ValueError("Cannot verify NiCr type/mass/EAM mapping in lammps.log.0; "
                                 "retain the complete log and review this engine version")
            for index, species in enumerate(words[4:], 1):
                if (labels[index] != species or species not in expected_mass
                        or not np.isfinite(masses[index])
                        or abs(masses[index] - expected_mass[species]) > 0.05):
                    raise ValueError(f"LAMMPS type {index}: label, mass and EAM element disagree; "
                                     "do not interpret this trajectory as the intended NiCr system")
            checked += 1
    if not checked:
        raise ValueError("No verifiable NiCr pair_coeff setup in lammps.log.0; mapping is unverified")
    return checked


def main(kind):
    parser = argparse.ArgumentParser(description=f"Read-only {kind} structure/progress and basin diagnostics")
    parser.add_argument("trajectory", type=Path)
    if kind == "fe":
        parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    try:
        directory = args.trajectory.resolve().parent
        config = sections(directory / "input.in")
        steps = config["control"].getint("n_steps")
        if steps < 1:
            raise ValueError("n_steps must be positive")
        data = progress(directory / "pykmc.out", steps)
        if kind == "fe":
            expected_cell = np.eye(3) * 42.8295
            expected_numbers = np.full(6746, 26)
            reference = read(args.reference)
            if (len(reference) != 6750 or not np.all(reference.numbers == 26)
                    or not np.isfinite(reference.positions).all() or not reference.pbc.all()
                    or not np.allclose(reference.cell.array, expected_cell, atol=1e-6, rtol=0)):
                raise ValueError("Expected the supplied 6750-site periodic Fe BCC reference")
            box = reference.cell.lengths()
            tree = cKDTree(reference.positions % box, boxsize=box)
            ideal = bulk("Fe", "bcc", a=2.8553, cubic=True).repeat((15, 15, 15))
            distance, sites = tree.query(ideal.positions % box)
            if distance.max() > 1e-6 or len(np.unique(sites)) != 6750:
                raise ValueError("Reference does not match the aligned ideal BCC lattice")
        else:
            expected_cell = np.diag([19.93475437521115, 19.93475437521115, 50.572])
            expected_numbers = np.array([24] * 87 + [28] * 360)
            print(f"INFO: atom_coloring_mode={config['atomicenvironment'].get('atom_coloring_mode', 'full')}")
            mapping_checks = verify_nicr_mapping(directory / "lammps.log.0")
        count = 0
        first_numbers = None
        for index, atoms in enumerate(iread(args.trajectory, index=":")):
            if (not np.array_equal(np.sort(atoms.numbers), expected_numbers)
                    or not np.isfinite(atoms.positions).all() or not atoms.pbc.all()
                    or not np.allclose(atoms.cell.array, expected_cell, atol=1e-6, rtol=0)):
                raise ValueError(f"Frame {index}: composition, finite positions or fixed cell/PBC do not match this example")
            if first_numbers is None:
                first_numbers = atoms.numbers.copy()
            elif not np.array_equal(atoms.numbers, first_numbers):
                raise ValueError(f"Frame {index}: atom-type order changed; review identity mapping before displacement analysis")
            if kind == "fe":
                _, sites = tree.query(atoms.positions % box)
                occupancy = np.bincount(sites, minlength=6750)
                if (occupancy == 0).sum() != 4 or np.maximum(occupancy - 1, 0).sum() != 0:
                    raise ValueError(f"Frame {index}: expected four vacant sites and zero excess atoms")
            count += 1
        if count != steps + 1:
            raise ValueError(f"Expected {steps + 1} frames; found {count}")
        # Establish completion/status before printing PASS.
        if "End of simulation" not in (directory / "pykmc.log").read_text():
            raise ValueError("Missing End of simulation marker")
        status = directory / "run-exit-status.txt"
        if status.exists() and status.read_text().strip() != "0":
            raise ValueError("Launcher exit status is not zero")
        print("PASS: saved structure and progress consistency")
        print(f"INFO: {count} frames, {steps} steps, final simulated time={data[-1, 2]:.6g} s")
        if kind == "fe":
            print("PASS: four vacant BCC sites and zero excess atoms in every saved frame")
        else:
            print(f"PASS: NiCr LAMMPS type/mass/EAM mapping ({mapping_checks} logged setups)")
            print("PASS: 360 Ni + 87 Cr with fixed slab cell; no WS vacancy count was performed")
        log_summary(directory, data)
    except (OSError, ValueError, KeyError, IndexError, configparser.Error) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
