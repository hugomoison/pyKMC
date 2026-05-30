# pyKMC Linux Installation Instructions

**Tested for:** Ubuntu/Debian and RHEL/CentOS-based systems
**Python:** 3.9–3.13
**LAMMPS:** `stable_22Jul2025_update3`

> **Tip:** For an automated install, run [`install_pykmc_linux.sh`](install_pykmc_linux.sh) instead of following the manual steps below — the script does everything automatically. See **Automated install** below for how to run it. To install manually, skip to [Section 0](#0-system-prerequisites).

---

## Automated install (recommended)

`install_pykmc_linux.sh` creates a `pykmc/` directory **inside your current working directory** and installs everything there. Choose where you want the install to live before running it.

1. Create (or choose) the folder where the install should live, and `cd` into it. Replace `/path/to/your/install-folder` with wherever you want the install to live (the script will create a `pykmc/` subfolder inside it):

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

To use a specific Python interpreter, set `PYTHON_BIN` before running:

```bash
PYTHON_BIN=/usr/bin/python3.12 /path/to/install_pykmc_linux.sh
```

When it finishes you'll have `pykmc/pykmc_env/`, `pykmc/lammps/`, `pykmc/IterativeRotationsAssignments/`, `pykmc/artn-plugin/`, and `pykmc/activate.sh` under the folder you chose.

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

```bash
module load StdEnv/2023
module load python/3.12.4
module load openmpi
module load mpi4py
module load cmake
```

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

On Linux with GNU `sed`:

```bash
sed -i 's/requires-python = "<3.13,>=3.9"/requires-python = "<3.14,>=3.9"/' pyKMC/pyproject.toml
```

---

## 3. Create virtual environment and install pyKMC

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

> **Note:** On DRAC clusters with the `mpi4py` module loaded, skip the `mpi4py` pip install.

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
  -DPKG_PLUGIN=on \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DPython_EXECUTABLE=$(which python)

make -j$(nproc)
make install-python

cd ../..
```

Create symlinks so pARTn can find CMake build outputs (pARTn's `configure` searches `lammps/src/` while CMake writes to `lammps/build/`):

```bash
mkdir -p lammps/src/styles
ln -sf ../../build/styles/lmpinstalledpkgs.h lammps/src/styles/lmpinstalledpkgs.h
ln -sf ../build/liblammps.so lammps/src/liblammps.so
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

> **Note:** On Linux, `.so` is the native shared library format, so no symlink workaround is needed (unlike macOS, which builds `.dylib`).

---

## 6. Build pARTn plugin

```bash
cd artn-plugin

./configure --with-lammps LAMMPS_PATH=$(pwd)/../lammps
```

Patch C++17 support (required by LAMMPS headers) using GNU `sed` (no `''` after `-i`, unlike macOS):

```bash
sed -i 's/^CXX=mpicxx$/CXX=mpicxx\nCXXFLAGS=-std=c++17/' make.inc
sed -i 's/$(CXX) -g -fPIC -I/$(CXX) -g -fPIC ${CXXFLAGS} -I/g' ENGINES/LAMMPS/Makefile
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

### Local Linux workstation

```bash
source pykmc_env/bin/activate
export LD_LIBRARY_PATH=$(pwd)/lammps/build:${LD_LIBRARY_PATH}
export PYTHONPATH=$(pwd)/artn-plugin/interface:$PYTHONPATH
export PYTHONPATH=$(pwd)/IterativeRotationsAssignments/interface:$PYTHONPATH

mpirun -n 8 python -m pykmc -in input.in
```

Or source the activation script created by the installer:

```bash
source activate.sh
```

### DRAC / Alliance HPC cluster (sbatch script)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048M

module load StdEnv/2023 python/3.12.4 openmpi mpi4py
source /home/$USER/pykmc/pykmc_env/bin/activate

export PYTHONPATH=$HOME/pykmc/artn-plugin/interface:$PYTHONPATH
export PYTHONPATH=$HOME/pykmc/IterativeRotationsAssignments/interface:$PYTHONPATH

srun --ntasks=$SLURM_NTASKS --distribution=block:block \
     --cpu-bind=cores --mem-bind=local \
     python -m pykmc -in input.in
```

### pARTn library path

In your `input.in`, set the pARTn library path (use the full absolute path):

```ini
[pARTn]
path_artnso = /full/path/to/pykmc/artn-plugin/lib/libartn-lmp.so
```

---

## Differences from macOS

| # | Topic | Linux | macOS |
|---|---|---|---|
| 1 | Shared libraries | `.so` natively | `.dylib` (symlinked to `.so`) |
| 2 | Library path variable | `LD_LIBRARY_PATH` | `DYLD_LIBRARY_PATH` |
| 3 | CPU count | `nproc` | `sysctl -n hw.ncpu` |
| 4 | `sed` syntax | `sed -i '...'` | `sed -i '' '...'` |
| 5 | IRA install | `pip install .` works directly | Pre-create `libira.so` symlink first |
| 6 | LAMMPS symlinks | Required so pARTn finds CMake outputs under `lammps/src/` (one `.so` link) | Required both for `.dylib` → `.so` *and* to bridge CMake outputs to `lammps/src/` |
| 7 | Package manager | `apt` / `dnf` | `brew` |
