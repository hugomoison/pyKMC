"""Read-only checks for a fresh, fixed-cell Ni FCC SIA tutorial trajectory."""
import argparse
import configparser
from pathlib import Path

import numpy as np
from ase.io import iread, read
from scipy.spatial import cKDTree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--progress', type=Path, default=Path('pykmc.out'))
    parser.add_argument('--input', type=Path, default=Path('input.in'))
    args = parser.parse_args()
    require(args.steps > 0, 'steps must be positive')
    ref = read(args.reference)
    require(len(ref) == 4000 and np.all(ref.numbers == 28), 'Expected 4000 Ni reference sites')
    cell = np.eye(3) * 35.2
    require(ref.pbc.all() and np.allclose(ref.cell.array, cell, atol=1e-6, rtol=0), 'Reference cell must be periodic cubic 35.2 A')
    require(np.isfinite(ref.positions).all(), 'Non-finite reference coordinates')
    tree = cKDTree(ref.positions % 35.2, boxsize=35.2)
    require(not tree.query_pairs(1e-6), 'Duplicate reference sites')
    count = 0
    for index, atoms in enumerate(iread(args.trajectory, index=':')):
        require(len(atoms) == 4001 and np.all(atoms.numbers == 28), f'Frame {index}: expected 4001 Ni atoms')
        require(np.isfinite(atoms.positions).all(), f'Frame {index}: non-finite coordinates')
        require(atoms.pbc.all() and np.allclose(atoms.cell.array, cell, atol=1e-6, rtol=0), f'Frame {index}: cell/PBC changed')
        _, sites = tree.query(atoms.positions % 35.2)
        occupancy = np.bincount(sites, minlength=4000)
        require(np.count_nonzero(occupancy == 0) == 0 and np.maximum(occupancy - 1, 0).sum() == 1,
                f'Frame {index}: expected zero vacancies and one excess atom; inspect structure/reference')
        count += 1
    require(count == args.steps + 1, f'Expected {args.steps + 1} frames, found {count}')
    print(f'PASS: {count} frames, 4001 Ni, fixed periodic cell; zero vacancies and one excess atom per frame')

    rows = []
    for line in args.progress.read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        expected = 3 if parts[0] == '0' else 10
        require(len(parts) == expected, 'Unexpected progress format; review the installed version before using this checker')
        rows.append([float(value) for value in parts])
    require(len(rows) == args.steps + 1, 'Unexpected number of progress rows')
    require([row[0] for row in rows] == list(range(args.steps + 1)), 'Progress steps are not 0..N')
    require(rows[0] == [0, 0, 0], 'Expected a fresh run starting at zero time')
    data = np.array(rows[1:])
    require(np.isfinite(data).all(), 'Non-finite progress values')
    require(np.all(data[:, 1] > 0) and np.all(np.diff(np.r_[0, data[:, 2]]) > 0), 'KMC time must increase')
    require(np.allclose(np.cumsum(data[:, 1]), data[:, 2], rtol=3e-6, atol=0), 'Cumulative time does not match step times')
    require(np.all(data[:, 5:7] > 0) and np.all(data[:, 6] >= data[:, 5]), 'Invalid event/total rate')
    print('PASS: progress rows, positive rates and increasing/cumulative physical time')

    cfg = configparser.ConfigParser()
    require(bool(cfg.read(args.input)), 'Input file not readable')
    sections = {name.lower(): cfg[name] for name in cfg.sections()}
    control = sections['control']
    require(control.getint('n_steps') == args.steps, 'Input n_steps differs from requested checks')
    require(not control.getboolean('basin', fallback=False), 'This checker is for the non-basin tutorial')
    rate = sections['rateconstant']
    require(rate.get('style', '').strip().lower() == 'constant', 'Rate check requires constant-prefactor model')
    prefactor, temperature = rate.getfloat('k0'), rate.getfloat('T')
    require(prefactor > 0 and temperature > 0, 'Prefactor and temperature must be positive')
    expected_rates = prefactor * np.exp(-data[:, 4] / (8.617333262e-5 * temperature))
    require(np.allclose(expected_rates, data[:, 5], rtol=1e-4, atol=0), 'Rates do not match input temperature/prefactor and barriers')
    print('PASS: selected-event rates agree with the Arrhenius expression')
    print('Selected reference IDs:', sorted({int(value) for value in data[:, 3]}))
    print(f'Selected barrier range: {data[:, 4].min():.6g} to {data[:, 4].max():.6g} eV')
    print('Checks are not proof of an exhaustive event catalogue or validated material kinetics.')


if __name__ == '__main__':
    main()
