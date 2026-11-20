"""Generate and check the aligned 4000-site FCC Ni reference for the SIA example."""
import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import iread, write
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('trajectory', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
reference = bulk('Ni', 'fcc', a=3.52, cubic=True).repeat((10, 10, 10))
box = reference.cell.lengths()
tree = cKDTree(reference.positions % box, boxsize=box)
count = 0
for frame, atoms in enumerate(iread(args.trajectory, index=':')):
    if len(atoms) != 4001 or not np.all(atoms.numbers == 28):
        raise ValueError('Expected 4001 Ni atoms per frame')
    if not atoms.pbc.all() or not np.allclose(atoms.cell.array, reference.cell.array, atol=1e-6, rtol=0):
        raise ValueError('Expected the aligned periodic 35.2 A cubic cell')
    if not np.isfinite(atoms.positions).all():
        raise ValueError('Non-finite coordinates')
    distances, sites = tree.query(atoms.positions % box)
    occupancy = np.bincount(sites, minlength=len(reference))
    vacancies = int(np.count_nonzero(occupancy == 0))
    excess = int(np.maximum(occupancy - 1, 0).sum())
    if vacancies != 0 or excess != 1:
        raise ValueError(f'Frame {frame}: unexpected occupancy; review reference alignment and structure')
    print(f'Frame {frame}: {vacancies} vacancies, {excess} excess atom')
    count += 1
if count == 0:
    raise ValueError('Empty trajectory')
with args.output.open('x') as handle:
    write(handle, reference, format='extxyz')
print(f'Wrote {args.output}: 4000 ideal sites; checked {count} trajectory frames')
print('Analysis reference only: do not replace initial_config.xyz with this file.')
