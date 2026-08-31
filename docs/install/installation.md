# Installation

pyKMC is a Python package; a working simulation also needs three other codes,
all currently required (no alternative implementations exist yet):

- **[LAMMPS](https://github.com/lammps/lammps)** — the energy/force engine
- **[pARTn](https://gitlab.com/mammasmias/artn-plugin)** — saddle-point searches (event discovery)
- **[IRA](https://github.com/mammasmias/IterativeRotationsAssignments)** — shape matching (event reconstruction)

The steps below assume you are comfortable building LAMMPS and managing Python
environments. If you would rather not build anything by hand, one-shot scripts
install the complete stack — see
[Automated install and platform guides](#automated-install-and-platform-guides)
at the end.

## 1. Python environment and pyKMC

Use a dedicated virtual environment (Python ≥ 3.10):

```bash
python3 -m venv /path/to/pykmc_env
source /path/to/pykmc_env/bin/activate
cd /path/to/pyKMC
pip install -e .
```

`mpi4py` must be linked against the same MPI you will run with. If
`from mpi4py import MPI` fails or the run crashes at startup, rebuild it from
source against your MPI compiler wrapper:

```bash
CC=mpicc pip install --no-binary mpi4py mpi4py --force-reinstall
```

On clusters that provide an `mpi4py` module, load the module instead of
pip-installing it.

## 2. LAMMPS

### Using an existing LAMMPS

You can use a LAMMPS you already have (a recent release; tested with
`stable_22Jul2025_update3`) provided it was built:

- as a **shared library** (`BUILD_SHARED_LIBS=on`),
- with **`PKG_PLUGIN`** (loads pARTn) and **`PKG_EXTRA-COMPUTE`** (CNA),
- with the pair-style packages your potentials need (e.g. `MANYBODY` for EAM),
- optionally **`PKG_PHONON`** — only needed for HTST rate prefactors
  (`dynamical_matrix`).

If so, the only step needed is installing its Python bindings into the active
venv:

```bash
cd /path/to/lammps/build      # the cmake build directory
make install-python
```

Verify the module and its packages:

```bash
python -c "import lammps; lmp = lammps.lammps(); print(lmp.installed_packages)"
```

### Building LAMMPS from scratch

Use cmake — the traditional make path no longer supports the PLUGIN package:

```bash
cd /path/to/lammps
mkdir build && cd build
cmake ../cmake \
  -DBUILD_SHARED_LIBS=on \
  -DLAMMPS_EXCEPTIONS=on \
  -DPKG_KSPACE=on -DPKG_MANYBODY=on -DPKG_MOLECULE=on -DPKG_RIGID=on \
  -DPKG_EXTRA-COMPUTE=on -DPKG_PLUGIN=on -DPKG_PHONON=on \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 \
  -DPython_EXECUTABLE=$(which python)
make -j$(nproc)
make install-python
```

More details in the [LAMMPS cmake guide](https://docs.lammps.org/Build_cmake.html).

## 3. pARTn

Build against your LAMMPS; this installs the `pypARTn` module into the active
venv:

```bash
cd /path/to/artn-plugin
cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=/path/to/lammps/build -DARTN_INSTALL_PYTHON=ON
cmake --build build && cmake --install build
```

With `pypARTn` installed this way, input files need no library-path
configuration (the `[pARTn]` section can stay empty) and no `PYTHONPATH`
exports.

More details in the [pARTn documentation](https://mammasmias.gitlab.io/artn-plugin/user_guide/Installation.html).

## 4. IRA

```bash
cd /path/to/ira
pip install .
```

More details in the [IRA documentation](https://mammasmias.github.io/IterativeRotationsAssignments/#compilation).

## 5. Verify

```bash
python -c "
from lammps import lammps
import ase, pykmc, ira_mod, pypARTn
print('All imports OK')
"
```

If an import fails, see [Troubleshooting](../troubleshooting.md).

## 6. Running

pyKMC runs under MPI. Use at least `n_sessions + 1` ranks: rank 0 runs the
main KMC loop and the remaining ranks are split among the `n_sessions` LAMMPS
instances (`[Control]` section of the input file).

```bash
mpirun -n 8 python -m pykmc -in input.in
```

On a cluster, submit through your scheduler as usual — e.g. with Slurm,
activate the venv in the job script and launch with `srun`:

```bash
source /path/to/pykmc_env/bin/activate
srun --ntasks=$SLURM_NTASKS python -m pykmc -in input.in
```

Build LAMMPS and pARTn with the same toolchain and MPI you run with, and
prefer your site's `mpi4py` module if one is provided.

## Automated install and platform guides

- **One-shot installers** build the whole stack (pyKMC + LAMMPS + pARTn + IRA)
  into a `pykmc_install/` folder under the current directory (deliberately not
  `pykmc`, which would shadow the package on import — see the platform guides):
  [`install_pykmc_linux.sh`](install_pykmc_linux.sh),
  [`install_pykmc_mac.sh`](install_pykmc_mac.sh)
- **Step-by-step platform walkthroughs** (system prerequisites, exact commands,
  troubleshooting): [Linux](pykmc_linux_Compile_Instructions.md),
  [macOS](pykmc_mac_Compile_Instructions.md)
- **DRAC/Alliance cluster specifics** (module loads, filesystem rules, sbatch
  templates):
  [cluster notes](pykmc_linux_Compile_Instructions.md#drac-alliance-hpc-clusters)
