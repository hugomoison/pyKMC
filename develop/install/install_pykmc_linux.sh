#!/bin/bash
#
# pyKMC Linux Installation Script
# Tested for: Ubuntu/Debian, RHEL/CentOS/Rocky
#
# Usage:
#   chmod +x install_pykmc_linux.sh
#   ./install_pykmc_linux.sh
#
# Override Python interpreter:
#   PYTHON_BIN=/usr/bin/python3.12 ./install_pykmc_linux.sh
#
# This script will create a "pykmc" directory in the current location
# and install everything inside it.
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

step() { echo -e "\n${YELLOW}========================================${NC}"; echo -e "${YELLOW}  $1${NC}"; echo -e "${YELLOW}========================================${NC}\n"; }
ok()   { echo -e "${GREEN}[OK] $1${NC}"; }
fail() { echo -e "${RED}[FAIL] $1${NC}"; exit 1; }

# ------------------------------------------
# 0a. Check prerequisites
# ------------------------------------------
step "Checking prerequisites"

command -v git >/dev/null 2>&1 || fail "git not found. Install with your package manager."

# Detect package manager and install missing dependencies
if command -v apt >/dev/null 2>&1; then
    PKGS=""
    dpkg -s build-essential >/dev/null 2>&1 || PKGS="$PKGS build-essential"
    dpkg -s gfortran        >/dev/null 2>&1 || PKGS="$PKGS gfortran"
    dpkg -s cmake           >/dev/null 2>&1 || PKGS="$PKGS cmake"
    dpkg -s libopenmpi-dev  >/dev/null 2>&1 || PKGS="$PKGS libopenmpi-dev openmpi-bin"
    dpkg -s libfftw3-dev    >/dev/null 2>&1 || PKGS="$PKGS libfftw3-dev"
    dpkg -s liblapack-dev   >/dev/null 2>&1 || PKGS="$PKGS liblapack-dev"
    dpkg -s python3-venv    >/dev/null 2>&1 || PKGS="$PKGS python3-venv"
    dpkg -s python3-dev     >/dev/null 2>&1 || PKGS="$PKGS python3-dev"
    if [ -n "$PKGS" ]; then
        echo "Installing missing packages:$PKGS"
        sudo apt update
        sudo apt install -y $PKGS
    fi
elif command -v dnf >/dev/null 2>&1; then
    PKGS=""
    rpm -q gcc-c++     >/dev/null 2>&1 || PKGS="$PKGS gcc-c++"
    rpm -q gcc-gfortran >/dev/null 2>&1 || PKGS="$PKGS gcc-gfortran"
    rpm -q cmake       >/dev/null 2>&1 || PKGS="$PKGS cmake"
    rpm -q openmpi-devel >/dev/null 2>&1 || PKGS="$PKGS openmpi-devel"
    rpm -q fftw-devel  >/dev/null 2>&1 || PKGS="$PKGS fftw-devel"
    rpm -q lapack-devel >/dev/null 2>&1 || PKGS="$PKGS lapack-devel"
    rpm -q python3-devel >/dev/null 2>&1 || PKGS="$PKGS python3-devel"
    if [ -n "$PKGS" ]; then
        echo "Installing missing packages:$PKGS"
        sudo dnf install -y $PKGS
    fi
    # RHEL-based distros often need MPI in PATH
    if [ -d /usr/lib64/openmpi/bin ] && ! command -v mpicc >/dev/null 2>&1; then
        export PATH=/usr/lib64/openmpi/bin:$PATH
        export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
    fi
else
    echo "Unknown package manager. Please install manually:"
    echo "  gcc, g++, gfortran, cmake, openmpi, fftw3, lapack, python3-dev"
fi

# Verify compilers are available
command -v gfortran >/dev/null 2>&1 || fail "gfortran not found"
command -v mpicc    >/dev/null 2>&1 || fail "mpicc not found"
command -v mpicxx   >/dev/null 2>&1 || fail "mpicxx not found"
command -v mpif90   >/dev/null 2>&1 || fail "mpif90 not found"
command -v cmake    >/dev/null 2>&1 || fail "cmake not found"
ok "All prerequisites found"

# ------------------------------------------
# 0b. Select a supported Python interpreter
# ------------------------------------------
step "Selecting Python interpreter"

find_supported_python() {
    for py in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$py" >/dev/null 2>&1; then
            local ver
            ver=$("$py" -c "import sys; print(sys.version_info.minor)")
            if [ "$ver" -ge 9 ] && [ "$ver" -le 13 ]; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(find_supported_python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    fail "No supported Python 3.9-3.13 found. Install one with your package manager."
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 13 ]; then
    fail "Supported Python range is 3.9-3.13, found $PYTHON_VERSION at $PYTHON_BIN"
fi

ok "Using Python $PYTHON_VERSION at $(command -v "$PYTHON_BIN")"

# ------------------------------------------
# 1. Create working directory and clone repos
# ------------------------------------------
step "Cloning repositories"

INSTALL_DIR="$(pwd)/pykmc"

if [ -d "$INSTALL_DIR" ]; then
    fail "Directory $INSTALL_DIR already exists. Remove it or run from a different location."
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

git clone -b develop https://github.com/hugomoison/pyKMC.git
git clone -b stable_22Jul2025_update3 --depth 1 https://github.com/lammps/lammps.git
git clone https://github.com/mammasmias/IterativeRotationsAssignments.git
git clone https://gitlab.com/mammasmias/artn-plugin.git
ok "All repositories cloned"

# ------------------------------------------
# 2. Fix Python version constraint (3.13 only)
# ------------------------------------------
step "Fixing Python version constraint"

if [ "$PYTHON_MINOR" -eq 13 ]; then
    sed -i 's/requires-python = "<3.13,>=3.9"/requires-python = "<3.14,>=3.9"/' pyKMC/pyproject.toml
    ok "Updated pyproject.toml for Python 3.13"
else
    ok "Python $PYTHON_VERSION is within range, no fix needed"
fi

# ------------------------------------------
# 3. Create virtual environment and install pyKMC
# ------------------------------------------
step "Creating virtual environment and installing pyKMC"

"$PYTHON_BIN" -m venv ./pykmc_env
source pykmc_env/bin/activate

python -m pip install --upgrade pip --quiet
python -m pip install -e ./pyKMC --quiet
ok "pyKMC installed"

# Rebuild mpi4py from source to match system MPI
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
python -m pip install --no-binary mpi4py mpi4py --force-reinstall --quiet
ok "mpi4py rebuilt from source"

# ------------------------------------------
# 4. Build LAMMPS
# ------------------------------------------
step "Building LAMMPS (this may take a few minutes)"

cd "$INSTALL_DIR/lammps"
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
  -DPython_EXECUTABLE="$(which python)" \
  > /dev/null 2>&1

make -j"$(nproc)" > /dev/null 2>&1
make install-python > /dev/null 2>&1

cd "$INSTALL_DIR"

# Symlinks for pARTn compatibility (pARTn configure expects LAMMPS libs/headers
# under lammps/src/, but CMake puts them under lammps/build/).
mkdir -p lammps/src/styles
ln -sf ../../build/styles/lmpinstalledpkgs.h lammps/src/styles/lmpinstalledpkgs.h
ln -sf ../build/liblammps.so lammps/src/liblammps.so

python -c "from lammps import lammps" || fail "LAMMPS Python bindings not working"
ok "LAMMPS built and installed"

# ------------------------------------------
# 5. Build IRA
# ------------------------------------------
step "Building IRA"

cd "$INSTALL_DIR/IterativeRotationsAssignments"
python -m pip install . --quiet

cd "$INSTALL_DIR"

python -c "import ira_mod" || fail "IRA not working"
ok "IRA built and installed"

# ------------------------------------------
# 6. Build pARTn plugin
# ------------------------------------------
step "Building pARTn plugin"

cd "$INSTALL_DIR/artn-plugin"

./configure --with-lammps LAMMPS_PATH="$INSTALL_DIR/lammps" > /dev/null 2>&1

# Patch C++17 support (required by LAMMPS headers)
sed -i 's/^CXX=mpicxx$/CXX=mpicxx\nCXXFLAGS=-std=c++17/' make.inc
sed -i 's/$(CXX) -g -fPIC -I/$(CXX) -g -fPIC ${CXXFLAGS} -I/g' ENGINES/LAMMPS/Makefile

make lmplib > /dev/null 2>&1

cd "$INSTALL_DIR"

# Install pypARTn Python interface
cp artn-plugin/interface/pypARTn.py "$(python -c 'import site; print(site.getsitepackages()[0])')/"

ls artn-plugin/lib/libartn-lmp.so > /dev/null || fail "pARTn library not found"
python -c "import pypARTn" || fail "pypARTn not working"
ok "pARTn built and installed"

# ------------------------------------------
# 7. Verify full installation
# ------------------------------------------
step "Verifying installation"

python -c "
from lammps import lammps
import ase, pykmc, ira_mod, pypARTn
print('All imports OK')
" || fail "Import verification failed"

ok "All components verified"

# ------------------------------------------
# 8. Create activation script
# ------------------------------------------
cat > "$INSTALL_DIR/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Source this file to activate the pyKMC environment:
#   source /path/to/pykmc/activate.sh

PYKMC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PYKMC_DIR/pykmc_env/bin/activate"
export LD_LIBRARY_PATH="$PYKMC_DIR/lammps/build:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PYKMC_DIR/artn-plugin/interface:${PYTHONPATH:-}"
export PYTHONPATH="$PYKMC_DIR/IterativeRotationsAssignments/interface:$PYTHONPATH"
echo "pyKMC environment activated. Run with:"
echo "  mpirun -n 8 python -m pykmc -in input.in"
ACTIVATE
chmod +x "$INSTALL_DIR/activate.sh"

# ------------------------------------------
# Done
# ------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To use pyKMC:"
echo "  source $INSTALL_DIR/activate.sh"
echo "  mpirun -n 8 python -m pykmc -in input.in"
echo ""
echo "In your input.in, set:"
echo "  [pARTn]"
echo "  path_artnso = $INSTALL_DIR/artn-plugin/lib/libartn-lmp.so"
echo ""
