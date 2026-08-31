# pyKMC Linux Installation Instructions

**Tested for:** Ubuntu/Debian and RHEL/CentOS-based systems
**Python:** 3.10–3.13
**LAMMPS:** `stable_22Jul2025_update3`

> **Tip:** For an automated install, run [`install_pykmc_linux.sh`](install_pykmc_linux.sh) instead of following the manual steps below — the script does everything automatically. See **Automated install** below for how to run it. To install manually, skip to [Section 0](#0-system-prerequisites).

---

## Automated install (recommended)

`install_pykmc_linux.sh` creates a `pykmc_install/` directory **inside your current working directory** and installs everything there. Choose where you want the install to live before running it.

1. Create (or choose) the folder where the install should live, and `cd` into it. Replace `/path/to/your/install-folder` with wherever you want the install to live (the script will create a `pykmc_install/` subfolder inside it):

   ```bash
   mkdir -p /path/to/your/install-folder
   cd /path/to/your/install-folder
   ```

2. Make the script executable (only required the first time):

   ```bash
   chmod +x /path/to/install_pykmc_linux.sh
   ```

3. Run it from the folder you chose in step 1:

   ```bash
   /path/to/install_pykmc_linux.sh
   ```

   To save a log for troubleshooting, tee the output:

   ```bash
   /path/to/install_pykmc_linux.sh 2>&1 | tee install.log
   ```

The script will prompt **once** for your `sudo` password so it can `apt`/`dnf` install any missing system packages (`build-essential`, `gfortran`, `cmake`, `libopenmpi-dev`, `openmpi-bin`, `libfftw3-dev`, `liblapack-dev`, `python3-venv`, `python3-dev` on Debian/Ubuntu). After that it runs unattended for roughly 10–20 minutes while LAMMPS compiles.

> **DRAC/Alliance HPC clusters:** there is **no** `sudo` prompt — the script auto-detects the
> cluster and skips the package-install stage entirely. Load the toolchain modules **before**
> running it (see the [cluster notes](#drac-alliance-hpc-clusters)), run it on a **login node**
> (compute nodes have no internet), and run it from a directory under `$SCRATCH`.

To use a specific Python interpreter, set `PYTHON_BIN` before running:

```bash
PYTHON_BIN=/usr/bin/python3.12 /path/to/install_pykmc_linux.sh
```

When it finishes you'll have `pykmc_install/pykmc_env/`, `pykmc_install/lammps/`, `pykmc_install/IterativeRotationsAssignments/`, `pykmc_install/artn-plugin/`, and `pykmc_install/activate.sh` under the folder you chose.

> The install root is **not** called `pykmc`: a directory of that name makes `import pykmc` resolve
> to it as an empty namespace package whenever Python runs from its parent, which fails later and
> misleadingly (`ImportError: cannot import name 'NeighborsList' from 'pykmc' (unknown location)`).
> The same applies if you follow the manual steps below — don't name your working directory `pykmc`.

---

## 0. System prerequisites

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y build-essential gfortran cmake git \
    libopenmpi-dev openmpi-bin libfftw3-dev liblapack-dev \
    python3 python3-venv python3-dev
```

### RHEL / CentOS / Rocky / Alma

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc-gfortran cmake git \
    openmpi openmpi-devel fftw-devel lapack-devel \
    python3 python3-devel

# Load MPI module (RHEL-based distros require this):
module load mpi/openmpi-x86_64
# or:
export PATH=/usr/lib64/openmpi/bin:$PATH
```

### DRAC / Alliance HPC clusters

> Validated end-to-end on **Trillium** (2026-06). Module versions below are Trillium's;
> check yours with `module spider <name>`.

```bash
module load StdEnv/2023
module load gcc/12.3        # must come first: openmpi/mpi4py are hidden until a compiler is loaded
module load openmpi/4.1.5
module load python/3.12.4
module load mpi4py/4.1.0    # pyKMC needs mpi4py >= 4.0.2 — mpi4py/4.0.0 is too old
module load cmake/3.31.0
```

Cluster ground rules (they shape every step below):

- **Login vs compute nodes:** login nodes have internet; compute nodes do **not**. All `git clone`
  and `pip` steps (including `make install-python`, which downloads build tools) must run on a
  **login node**. Only the pure compile (`make` / `cmake --build`) benefits from a compute node.
- **Filesystems:** install under **`$SCRATCH`** on Trillium — `$HOME` and `$PROJECT` are read-only
  from compute jobs, and jobs must also *run* from `$SCRATCH`.
- **Fresh job shells are empty:** `salloc`/`debugjob` shells start with no modules, no venv, no
  exported variables. Re-run the module loads + `source .../activate` in every new job shell.
- **Accounts differ per cluster:** e.g. `rrg-xxxx_cpu` on Narval/Beluga but plain `rrg-xxxx` on
  Trillium. Check yours with `sshare -U $USER`.
- **Do not test MPI imports bare on a compute node:** `python -c "from mpi4py import MPI"` (or
  creating a `lammps()` object) hangs in an allocation unless launched via `srun --ntasks=1 …`.
  On login nodes it works as a singleton.

### Verify compilers

```bash
gfortran --version
mpicc --version
mpicxx --version
cmake --version
```

---

## 1. Clone repositories

```bash
mkdir pykmc_install && cd pykmc_install   # any name but "pykmc" — see the namespace-package note above

git clone -b develop https://github.com/hugomoison/pyKMC.git
git clone -b stable_22Jul2025_update3 --depth 1 https://github.com/lammps/lammps.git
git clone https://github.com/mammasmias/IterativeRotationsAssignments.git
git clone https://gitlab.com/mammasmias/artn-plugin.git
```

> **Note:** pyKMC **must** use the `develop` branch.

---

## 2. Python version

pyKMC requires **Python ≥ 3.10** (`requires-python = ">=3.10"` in
`pyKMC/pyproject.toml`), with no upper bound — Python 3.13 installs without any
edit. If one of the dependencies does not yet ship a wheel for your Python
version, fall back to the most recent version that does (3.12 is a safe choice).

---

## 3. Create virtual environment and install pyKMC

> **DRAC/Alliance clusters: stop — don't run the default commands below.** Load the toolchain
> modules first (Section 0), then follow the cluster pattern in the callout at the end of this
> section instead: the venv must be created with `virtualenv --no-download` **after** the
> `mpi4py` module is loaded, and mpi4py must **not** be pip-installed.

```bash
python3 -m venv ./pykmc_env
source pykmc_env/bin/activate
```

After activation, use bare `python` (the venv resolves the correct version):

```bash
python -m pip install --upgrade pip
python -m pip install -e ./pyKMC
```

Rebuild `mpi4py` from source to match your system MPI:

```bash
export CC=mpicc CXX=mpicxx FC=mpif90
python -m pip install --no-binary mpi4py mpi4py --force-reinstall
```

> **DRAC/Alliance clusters:** do **not** pip-install mpi4py — the `mpi4py` module provides an
> ABI-matched build (the Alliance wheelhouse deliberately has no real mpi4py wheel). Load the
> module **before** creating the venv, and create the venv with `virtualenv --no-download` so
> the module's mpi4py is visible inside it:
>
> ```bash
> virtualenv --no-download ./pykmc_env
> source pykmc_env/bin/activate
> python -m pip install --no-index --upgrade pip
> python -c "from mpi4py import MPI; print(MPI.Get_library_version())"   # login node only
> ```
>
> The check must print the Open MPI version of the loaded `openmpi` module (e.g. 4.1.5).
> If it shows a different MPI, the venv was created before the module load — recreate it.
> `pip install -e ./pyKMC` then resolves most dependencies from the Alliance wheelhouse and
> reports mpi4py as already satisfied by the module.

---

## 4. Build LAMMPS

```bash
cd lammps
mkdir build && cd build

cmake ../cmake \
  -DBUILD_SHARED_LIBS=on \
  -DLAMMPS_EXCEPTIONS=on \
  -DPKG_BASIC=on \
  -DPKG_KSPACE=on \
  -DPKG_MANYBODY=on \
  -DPKG_RIGID=on \
  -DPKG_MOLECULE=on \
  -DPKG_EXTRA-COMPUTE=on \
  -DPKG_PHONON=on \
  -DPKG_PLUGIN=on \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DPython_EXECUTABLE=$(which python)

make -j$(nproc)
python -m pip uninstall -y lammps   # drop the PyPI wheel pulled in by pyKMC's dependencies
make install-python

cd ../..
```

> `PKG_PHONON` provides the `dynamical_matrix` command required by the HTST rate-constant
> prefactor — without it HTST silently falls back to the constant `k0`.

> `pip install -e ./pyKMC` (step 3) pulls a generic `lammps` wheel from PyPI to satisfy the
> `lammps>=…` dependency. That wheel has no PLUGIN/PHONON support — the `pip uninstall` above
> removes it so `make install-python` installs the wheel built from **this** LAMMPS.

> **DRAC/Alliance clusters:** run the compile (`make -j`) on a compute node if you like
> (`debugjob` gives a full node), but **`make install-python` must run on a login node** — it
> creates an isolated build environment and downloads `build`/`wheel` from PyPI, which fails on
> compute nodes (`Network is unreachable` after long retries that look like a hang). The build
> directory under `$SCRATCH` is shared, so compiling on a compute node and then running
> `make install-python` from a login shell works seamlessly.

Verify:

```bash
python -c "from lammps import lammps; print('LAMMPS OK')"
```

---

## 5. Build IRA

```bash
cd IterativeRotationsAssignments
python -m pip install .
cd ..
```

Verify:

```bash
python -c "import ira_mod; print('IRA OK')"
```

> `import pykmc` itself requires `ira_mod` (imported at package top level), so install IRA
> **before** testing any pyKMC import.

---

## 6. Build pARTn plugin

The following will directly install the `pypARTn` python module into the venv path. If you need a custom location for the package, specify additional `-DCMAKE_INSTALL_PREFIX=<your/custom/path>`.
```bash
cd artn-plugin
cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=$(pwd)/../lammps/build -DARTN_INSTALL_PYTHON=ON \
      -DCMAKE_CXX_FLAGS_INIT="-std=c++17"
cmake --build build
cmake --install build
```

> With `pypARTn` installed into the venv this way, pyKMC input files need **no `path_artnso`**
> (the `[pARTn]` section can stay empty) and **no `PYTHONPATH` exports** — both `pypARTn` and
> `ira_mod` are resolved from the venv. If you are migrating from an older PYTHONPATH-based
> install, drop the old `export PYTHONPATH=…/artn-plugin/interface` and
> `…/IterativeRotationsAssignments/interface` lines: they would shadow the venv packages.

Verify (under `mpirun -np 1`: instantiating pARTn/LAMMPS from bare `python` uses MPI
"singleton" init, which can crash the terminal on OpenMPI 5 systems such as Ubuntu 26.04):

```bash
mpirun -np 1 python -c "import pypARTn; a=pypARTn.artn(engine='lmp'); print('pypARTn OK')"
```

---

## 7. Verify installation

```bash
source pykmc_env/bin/activate

python -c "
from lammps import lammps
import ase, pykmc, ira_mod, pypARTn
print('All imports OK')
"
```

---

## 8. Running pyKMC

### Local Linux workstation

```bash
source pykmc_env/bin/activate
export LD_LIBRARY_PATH=$(pwd)/lammps/build:${LD_LIBRARY_PATH}

mpirun -n 8 python -m pykmc -in input.in
```

Or source the activation script created by the installer:

```bash
source activate.sh
```

### DRAC / Alliance HPC cluster (sbatch script)

The Slurm geometry differs per cluster:

**Trillium (whole-node scheduling — no partial nodes, no `--mem*` flags).**
The install was validated on Trillium (2026-06) with an 8-task smoke test
(`srun --ntasks=8 --cpu-bind=cores python -m pykmc -in input.in` for `n_sessions = 7`);
the template below scales the same pattern to a full node:

```bash
#!/bin/bash
#SBATCH --account=rrg-xxxx           # no _cpu suffix on Trillium; check `sshare -U $USER`
#SBATCH --nodes=1                    # whole node = 192 cores
#SBATCH --ntasks-per-node=192
#SBATCH --time=01:00:00

module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0
source "$SCRATCH/pykmc_install/pykmc_env/bin/activate"
export LD_LIBRARY_PATH="$SCRATCH/pykmc_install/lammps/build:${LD_LIBRARY_PATH:-}"

cd "$SCRATCH/your_run_dir"           # run from $SCRATCH — $HOME/$PROJECT are read-only in jobs

srun --ntasks=$SLURM_NTASKS --distribution=block:block \
     --cpu-bind=cores --mem-bind=local \
     python -m pykmc -in input.in
```

Set `n_sessions` (the number of parallel LAMMPS instances, `[Control]` section) to at
most `ntasks − 1`: rank 0 runs the main KMC loop and the remaining ranks are split
among the instances. With `n_sessions = ntasks − 1` each instance gets a single rank
(e.g. the example input with `n_sessions = 7` runs as `srun --ntasks=8`; a full
Trillium node supports up to `n_sessions = 191`).

**Narval / Beluga (per-core scheduling)** — the pattern used by existing production
sbatch scripts (not re-exercised in the Trillium validation):

```bash
#!/bin/bash
#SBATCH --account=rrg-xxxx_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048M

module load StdEnv/2023 python/3.12.4 openmpi mpi4py
source /home/$USER/pykmc_install/pykmc_env/bin/activate

srun --ntasks=$SLURM_NTASKS --distribution=block:block \
     --cpu-bind=cores --mem-bind=local \
     python -m pykmc -in input.in
```

---

## Differences from macOS

| # | Topic | Linux | macOS |
|---|---|---|---|
| 1 | Shared libraries | `.so` natively | `.dylib` (symlinked to `.so`) |
| 2 | Library path variable | `LD_LIBRARY_PATH` | `DYLD_LIBRARY_PATH` |
| 3 | CPU count | `nproc` | `sysctl -n hw.ncpu` |
| 4 | `sed` syntax | `sed -i '...'` | `sed -i '' '...'` |
| 5 | IRA install | `pip install .` works directly | `pip install .` works directly |
| 6 | Package manager | `apt` / `dnf` | `brew` |
