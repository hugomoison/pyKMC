"""Read-only checks for a fresh, fixed-cell Ni vacancy+SIA tutorial run.

Accepts a completed step budget or a structurally confirmed all-crystalline exit.
Requires NumPy, SciPy and ASE from the simulation environment. This is not a
general trajectory validator or a test of event-catalogue completeness.
"""
import argparse
from collections import Counter
import configparser
from pathlib import Path
import sys

import numpy as np
from ase.io import iread, read
from scipy.spatial import cKDTree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(args):
    cfg = configparser.ConfigParser()
    require(bool(cfg.read(args.input)), 'Input file not readable')
    sections = {name.lower(): cfg[name] for name in cfg.sections()}
    control, rate = sections['control'], sections['rateconstant']
    budget = control.getint('n_steps')
    require(budget > 0, 'n_steps must be positive')
    require(not control.getboolean('basin', fallback=False), 'This checker is for the non-basin tutorial')
    for name in ('restart_file', 'reference_table', 'visited_environments'):
        require(control.get(name, '').strip().lower() in ('', 'none'), f'{name}: use a fresh run for this checker')
    require(rate.get('style', '').strip().lower() == 'constant', 'Expected constant-prefactor rates')
    k0, temperature = rate.getfloat('k0'), rate.getfloat('T')
    require(np.isfinite([k0, temperature]).all() and k0 > 0 and temperature > 0,
            'Temperature and prefactor must be finite and positive')

    cell = np.eye(3) * 35.2
    ref = read(args.reference)
    require(len(ref) == 4000 and np.all(ref.numbers == 28), 'Expected 4000 Ni reference sites')
    require(ref.pbc.all() and np.allclose(ref.cell.array, cell, atol=1e-6, rtol=0),
            'Reference must have a periodic cubic 35.2 A cell')
    require(np.isfinite(ref.positions).all(), 'Non-finite reference coordinates')
    reference_positions = ref.positions % 35.2
    tree = cKDTree(reference_positions, boxsize=35.2)
    require(not tree.query_pairs(1e-6), 'Duplicate reference sites')
    half_grid = reference_positions / 1.76
    require(np.allclose(half_grid, np.round(half_grid), atol=1e-6, rtol=0)
            and np.all(np.round(half_grid).astype(int).sum(axis=1) % 2 == 0),
            'Reference is not the aligned FCC lattice expected by this tutorial')

    records = []
    for index, atoms in enumerate(iread(args.trajectory, index=':')):
        require(index <= budget, 'Trajectory has more frames than the input step budget')
        require(len(atoms) == 4000 and np.all(atoms.numbers == 28), f'Frame {index}: expected 4000 Ni atoms')
        require(np.isfinite(atoms.positions).all(), f'Frame {index}: non-finite coordinates')
        require(atoms.pbc.all() and np.allclose(atoms.cell.array, cell, atol=1e-6, rtol=0),
                f'Frame {index}: cell/PBC differs from reference')
        _, assignments = tree.query(atoms.positions % 35.2)
        occupancy = np.bincount(assignments, minlength=4000)
        pattern = (int(np.count_nonzero(occupancy == 0)), int(np.maximum(occupancy-1, 0).sum()))
        require(pattern in ((1, 1), (0, 0)),
                f'Frame {index}: unexpected vacancy/interstitial counts {pattern}; inspect reference and structure')
        records.append(pattern)
    require(len(records) >= 2, 'Expected an initial frame and at least one executed event')
    require(records[0] == (1, 1), 'Initial frame must contain one vacancy and one excess atom')
    completed = len(records) - 1
    perfect = [i for i, pattern in enumerate(records) if pattern == (0, 0)]
    require(not perfect or perfect == [completed],
            'Defect-free frame before the end: inspect the trajectory and version-specific stopping behavior')

    rows = []
    for line in args.progress.read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        require(len(parts) == (3 if parts[0] == '0' else 10), 'Unexpected progress format; review installed version')
        rows.append([float(x) for x in parts])
    require(len(rows) == len(records), 'Progress/trajectory lengths differ')
    require([r[0] for r in rows] == list(range(completed+1)), 'Progress steps must be consecutive from 0')
    require(rows[0] == [0, 0, 0], 'Expected zero initial time in a fresh run')
    data = np.array(rows[1:])
    require(np.isfinite(data).all(), 'Non-finite progress values')
    require(np.all(data[:, 1] > 0) and np.all(data[:, 2] > 0), 'Waiting and cumulative times must be positive')
    require(np.all(np.diff(data[:, 2]) >= 0), 'Printed cumulative time decreased')
    require(np.allclose(np.cumsum(data[:, 1]), data[:, 2], rtol=3e-6, atol=0),
            'Cumulative times do not match waiting times within output rounding')
    require(np.all(data[:, 4] >= 0) and np.all(data[:, 5:7] > 0)
            and np.all(data[:, 6] >= data[:, 5]*(1-3e-6)), 'Invalid barriers or event/total rates')
    require(np.all(data[:, 3] >= 0) and np.all(data[:, 3] == np.round(data[:, 3])), 'Invalid reference-event ID')
    expected = k0*np.exp(-data[:, 4]/(8.617333262e-5*temperature))
    require(np.allclose(data[:, 5], expected, rtol=1e-4, atol=0), 'Rates disagree with the input Arrhenius model')
    log = args.log.read_text()
    require('End of simulation' in log, 'Missing End of simulation; check launcher/scheduler status')
    crystal_stop = 'Only atoms with cristalline environment' in log
    require(not crystal_stop or bool(perfect), 'All-crystalline log message conflicts with WS counts')
    require(completed == budget or (perfect and crystal_stop),
            'Early stop without confirmed recombination/all-crystalline message')

    print('PASS: structure, progress and selected-event rate checks')
    print(f'Completed {completed} of {budget} requested steps; {len(records)} frames, 4000 Ni, fixed periodic cell')
    print('WS frame counts (vacancies, excess atoms):', dict(Counter(records)))
    if perfect:
        print('OUTCOME: recombination observed; inspect the reported transition in OVITO')
        print(f'Transition: frame {completed-1} -> {completed}; 1+1 defects -> 0+0')
        print(f'Selected reference ID: {int(data[-1,3])}; barrier: {data[-1,4]:.6g} eV')
        if completed > 1:
            change = data[-1, 7]-data[-2, 7]
            print(f'Final potential-energy change from printed output: {change:.6g} eV')
            if change >= 0:
                print('WARNING: no final energy decrease; review the structures and energy record')
    else:
        print('OUTCOME: step budget reached; recombination not observed')
    repeated = int(np.count_nonzero(np.diff(data[:, 2]) == 0))
    if repeated:
        print(f'NOTE: {repeated} repeated printed T values; positive dT and cumulative sums checked within rounding')
    unusual = data[data[:, 1] > 1e6*np.median(data[:, 1])]
    if len(unusual):
        print('WARNING: unusually large waiting time(s) relative to this run; not a diagnosis:')
        for row in unusual[:10]:
            print(f'  step {int(row[0])}: dT={row[1]:.6g} s, Ea={row[4]:.6g} eV, k_tot={row[6]:.6g} ps^-1')
        print('Inspect candidate channels before/at/after these steps in pykmc.events and pykmc.info')
    print('Structural recombination does not validate the event catalogue or physical recombination lifetime.')
    print('Also confirm local launcher exit status or Slurm completion independently.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--input', type=Path, default=Path('input.in'))
    parser.add_argument('--progress', type=Path, default=Path('pykmc.out'))
    parser.add_argument('--log', type=Path, default=Path('pykmc.log'))
    args = parser.parse_args()
    try:
        check(args)
    except (ValueError, OSError, KeyError, configparser.Error) as error:
        print(f'CHECK FAILED: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
