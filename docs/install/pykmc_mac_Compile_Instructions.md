# pyKMC macOS Installation Instructions

**Tested on:** macOS (Apple Silicon M-series), March 2026
**Python:** 3.10–3.13
**LAMMPS:** `stable_22Jul2025_update3`

> **Tip:** For an automated install, run [`install_pykmc_mac.sh`](install_pykmc_mac.sh) instead of following the manual steps below — the script does everything automatically. See **Automated install** below for how to run it. To install manually, skip to [Section 0](#0-system-prerequisites).

---

## Automated install (recommended)

`install_pykmc_mac.sh` creates a `pykmc_install/` directory **inside your current working directory** and installs everything there. Choose where you want the install to live before running it.

Before running, make sure Xcode Command Line Tools and Homebrew are installed (the script checks for them and exits with a message if either is missing):

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"   # Apple Silicon
```

Then:

1. Create (or choose) the folder where the install should live, and `cd` into it. Replace `/path/to/your/install-folder` with wherever you want the install to live (the script will create a `pykmc_install/` subfolder inside it):

   ```bash
   mkdir -p /path/to/your/install-folder
   cd /path/to/your/install-folder
   ```

2. Make the script executable (only required the first time):

   ```bash
   chmod +x /path/to/install_pykmc_mac.sh
   ```

3. Run it from the folder you chose in step 1:

   ```bash
   /path/to/install_pykmc_mac.sh
   ```

   To save a log for troubleshooting, tee the output:

   ```bash
   /path/to/install_pykmc_mac.sh 2>&1 | tee install.log
   ```

The script uses Homebrew to install any missing packages (`gcc`, `cmake`, `fftw`, `open-mpi`), so no `sudo` is required. It then runs unattended for roughly 10–20 minutes while LAMMPS compiles.

To use a specific Python interpreter, set `PYTHON_BIN` before running:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3.12 /path/to/install_pykmc_mac.sh
```

When it finishes you'll have `pykmc_install/pyKMC/`, `pykmc_install/pykmc_env/`, `pykmc_install/lammps/`, `pykmc_install/IterativeRotationsAssignments/`, `pykmc_install/artn-plugin/`, and `pykmc_install/activate.sh` under the folder you chose.

> The install root is **not** called `pykmc`: a directory of that name makes `import pykmc` resolve
> to it as an empty namespace package whenever Python runs from its parent, which fails later and
> misleadingly (`ImportError: cannot import name 'NeighborsList' from 'pykmc' (unknown location)`).
> The same applies if you follow the manual steps below — don't name your working directory `pykmc`.

---

## 0. System prerequisites

Install Xcode Command Line Tools (C/C++ compilers, `git`, `make`):

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add Homebrew to `PATH` (Apple Silicon):

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Install required packages:

```bash
brew install gcc open-mpi cmake fftw
```

Python 3.10–3.13 is required. If you don't already have one:

```bash
brew install python@3.13
```

Verify compilers:

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

Use a specific Python version for venv creation (e.g. `python3.13`):

```bash
python3.13 -m venv ./pykmc_env
source pykmc_env/bin/activate
```

After activation, use bare `python` (the venv resolves the correct version):

```bash
python -m pip install --upgrade pip
python -m pip install -e ./pyKMC
```

Rebuild `mpi4py` from source to match your local OpenMPI (the pip binary wheel causes segfaults with `mpirun`):

```bash
export CC=mpicc CXX=mpicxx FC=mpif90
python -m pip install --no-binary mpi4py mpi4py --force-reinstall
```

---

## 4. Build LAMMPS

Builds inside `lammps/build/`. Python bindings go into the active venv only.

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

make -j$(sysctl -n hw.ncpu)

# Drop any lammps wheel pip pulled from PyPI, so the wheel built from THIS LAMMPS
# is the one left in the venv
python -m pip uninstall -y lammps
make install-python

cd ../..
```

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

Verify (under `mpirun -np 1`: instantiating pARTn/LAMMPS from bare `python` uses MPI
"singleton" init, which can crash the terminal on some OpenMPI 5 systems):

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

Every time you run pyKMC, activate the environment:

```bash
source pykmc_env/bin/activate
```

Or simply source the activation script created by the installer, which does the same
and also drops any inherited `PYTHONPATH`:

```bash
source activate.sh
```

> No `DYLD_LIBRARY_PATH` export is needed: on a clean Apple Silicon install both
> `liblammps.dylib` and `libartn-lmp.dylib` resolve through their install names, and
> `activate.sh` sets no such variable. If a dylib does fail to load on your machine,
> fall back to `export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:${DYLD_LIBRARY_PATH}"`.

Run with MPI — use at least `n_sessions + 1` ranks: rank 0 runs the main KMC loop and
the remaining ranks are split among the `n_sessions` LAMMPS instances (`[Control]`
section of the input file):

```bash
mpirun -n 8 python -m pykmc -in input.in
```
