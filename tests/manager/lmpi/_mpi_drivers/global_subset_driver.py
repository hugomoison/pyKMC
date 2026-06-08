"""MPI body for the global-subset regression test.

Launched as:
    mpirun -n <nproc> python global_subset_driver.py <n_sessions> <n_global_sessions> <ini> [use_rank_0]

Builds an FCC Ni system, runs the production kmc.py mode sequence with the global
LAMMPS restricted to the first <n_global_sessions> workers, and prints on rank 0:
    GLOBAL_RANKS=[...]
    GLOBAL_PE=<value>
Exit 0 on success. A deadlock is caught by the parent process's timeout.
"""
import sys

import numpy as np
from mpi4py import MPI

from pykmc import Config, System
from pykmc.enginemanager.lmpi.pool import ManagerFactory

n_sessions = int(sys.argv[1])
n_global_sessions = int(sys.argv[2])
ini_path = sys.argv[3]
use_rank_0 = (len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "yes"))

# --- FCC Ni system (matches tests/conftest.py system_single_type_fcc) ---
a, repeat = 3.52, 4
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
positions = []
for i in range(repeat):
    for j in range(repeat):
        for k in range(repeat):
            for at in basis:
                positions.append(at + np.array([i, j, k]) * a)
system = System()
system.positions = np.array(positions)
system.types = ["Ni"] * len(system.positions)
system.cell = np.array([[repeat * a, 0, 0], [0, repeat * a, 0], [0, 0, repeat * a]])
system.pbc = np.array([True, True, True])
system.index = np.arange(len(system.positions))

config = Config.from_ini_file(ini_path)

factory = ManagerFactory(
    n_sessions=n_sessions,
    use_rank_0=use_rank_0,
    n_global_sessions=n_global_sessions,
)
manager = factory.launch()
if manager is None:
    sys.exit(0)  # engine ranks return here

# Production kmc.py rhythm: init -> local phase -> global phase -> local -> close
manager.initialize_sessions(config, system)
manager.use_local()
manager.set_all_positions(system.positions)
manager.use_global()
pe = manager.global_get_potential_energy()
manager.use_local()
manager.close_all()

print(f"GLOBAL_RANKS={factory.global_ranks}", flush=True)
print(f"GLOBAL_PE={pe}", flush=True)
sys.exit(0)
