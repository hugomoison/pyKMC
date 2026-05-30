# pyKMC macOS Installation Instructions

**Tested on:** macOS (Apple Silicon M-series), March 2026
**Python:** 3.9–3.13
**LAMMPS:** `stable_22Jul2025_update3`

> **Tip:** For an automated install, run [`install_pykmc_mac.sh`](install_pykmc_mac.sh) instead of following the manual steps below — the script does everything automatically. See **Automated install** below for how to run it. To install manually, skip to [Section 0](#0-system-prerequisites).

---

## Automated install (recommended)

`install_pykmc_mac.sh` creates a `pykmc/` directory **inside your current working directory** and installs everything there. Choose where you want the install to live before running it.

Before running, make sure Xcode Command Line Tools and Homebrew are installed (the script checks for them and exits with a message if either is missing):

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"   # Apple Silicon
```

Then:

1. Create (or choose) the folder where the install should live, and `cd` into it. Replace `/path/to/your/install-folder` with wherever you want the install to live (the script will create a `pykmc/` subfolder inside it):

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

When it finishes you'll have `pykmc/pykmc_env/`, `pykmc/lammps/`, `pykmc/IterativeRotationsAssignments/`, `pykmc/artn-plugin/`, and `pykmc/activate.sh` under the folder you chose.

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

Python 3.9–3.13 is required. If you don't already have one:

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
mkdir pykmc && cd pykmc

git clone -b develop https://github.com/hugomoison/pyKMC.git
git clone -b stable_22Jul2025_update3 --depth 1 https://github.com/lammps/lammps.git
git clone https://github.com/mammasmias/IterativeRotationsAssignments.git
git clone https://gitlab.com/mammasmias/artn-plugin.git
```

> **Note:** pyKMC **must** use the `develop` branch.

---

## 2. Fix Python version constraint

Only needed if using Python 3.13. Edit `pyKMC/pyproject.toml`:

- **Change:** `requires-python = "<3.13,>=3.9"`
- **To:** `requires-python = "<3.14,>=3.9"`

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
  -DPKG_PLUGIN=on \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DPython_EXECUTABLE=$(which python)

make -j$(sysctl -n hw.ncpu)
make install-python

cd ../..
```

Create symlinks so pARTn can find CMake build outputs:

```bash
mkdir -p lammps/src/styles
ln -sf ../../build/styles/lmpinstalledpkgs.h lammps/src/styles/lmpinstalledpkgs.h
ln -sf ../build/liblammps.dylib lammps/src/liblammps.dylib
ln -sf ../build/liblammps.dylib lammps/src/liblammps.so
```

Verify:

```bash
python -c "from lammps import lammps; print('LAMMPS OK')"
```

---

## 5. Build IRA

macOS builds `.dylib`, not `.so` — pre-create a symlink so `pip install` works the first time.

```bash
cd IterativeRotationsAssignments
mkdir -p lib
ln -sf libira.dylib lib/libira.so
python -m pip install .
cd ..
```

Verify:

```bash
python -c "import ira_mod; print('IRA OK')"
```

---

## 6. Build pARTn plugin

```bash
cd artn-plugin

./configure --with-lammps LAMMPS_PATH=$(pwd)/../lammps
```

Patch C++17 support (required by LAMMPS headers):

```bash
sed -i '' 's/^CXX=mpicxx$/CXX=mpicxx\nCXXFLAGS=-std=c++17/' make.inc
sed -i '' 's/$(CXX) -g -fPIC -I/$(CXX) -g -fPIC ${CXXFLAGS} -I/g' ENGINES/LAMMPS/Makefile
```

Build the plugin:

```bash
make lmplib
cd ..
```

Install the `pypARTn` Python interface into the venv:

```bash
cp artn-plugin/interface/pypARTn.py "$(python -c 'import site; print(site.getsitepackages()[0])')/"
```

Verify:

```bash
ls artn-plugin/lib/libartn-lmp.so
python -c "import pypARTn; print('pypARTn OK')"
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

Every time you run pyKMC, activate the environment and set these paths:

```bash
source pykmc_env/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:${DYLD_LIBRARY_PATH}"
export PYTHONPATH=$(pwd)/artn-plugin/interface:$PYTHONPATH
export PYTHONPATH=$(pwd)/IterativeRotationsAssignments/interface:$PYTHONPATH
```

Or simply source the activation script created by the installer:

```bash
source activate.sh
```

Run with MPI (`n = n_sessions + 1` when `engine_use_rank_0 = False`):

```bash
mpirun -n 8 python -m pykmc -in input.in
```

In your `input.in`, set the pARTn library path (use the full absolute path):

```ini
[pARTn]
path_artnso = /full/path/to/pykmc/artn-plugin/lib/libartn-lmp.so
```
