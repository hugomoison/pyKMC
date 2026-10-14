#!/bin/bash
#
# pyKMC macOS Installation Script
# Tested on: macOS (Apple Silicon), March 2026
#
# Usage:
#   chmod +x install_pykmc_mac.sh
#   ./install_pykmc_mac.sh
#
# Override Python interpreter:
#   PYTHON_BIN=/opt/homebrew/bin/python3.12 ./install_pykmc_mac.sh
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

command -v brew >/dev/null 2>&1 || fail "Homebrew not found. Install from https://brew.sh"
command -v git  >/dev/null 2>&1 || fail "git not found. Run: xcode-select --install"
ok "Homebrew and git found"

# Install missing Homebrew packages (skip upgrades)
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_UPGRADE=1
MISSING=""
for pkg in gcc cmake fftw open-mpi; do
    brew list "$pkg" >/dev/null 2>&1 || MISSING="$MISSING $pkg"
done
if [ -n "$MISSING" ]; then
    echo "Installing missing packages:$MISSING"
    brew install $MISSING
fi

# Verify compilers are available
command -v gfortran >/dev/null 2>&1 || fail "gfortran not found. Install with: brew install gcc"
command -v mpicc    >/dev/null 2>&1 || fail "mpicc not found. Install with: brew install open-mpi"
command -v mpicxx   >/dev/null 2>&1 || fail "mpicxx not found. Install with: brew install open-mpi"
command -v mpif90   >/dev/null 2>&1 || fail "mpif90 not found. Install with: brew install open-mpi"
ok "All prerequisites found"

# ------------------------------------------
# 0b. Select a supported Python interpreter
# ------------------------------------------
step "Selecting Python interpreter"

find_supported_python() {
    for py in python3.13 python3.12 python3.11 python3.10 python3.9; do
        if command -v "$py" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(find_supported_python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    echo "No supported Python 3.9-3.13 found. Installing python@3.13 with Homebrew..."
    brew install python@3.13
    PYTHON_BIN="$(brew --prefix python@3.13)/bin/python3.13"
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
    sed -i '' 's/<3.13,/<3.14,/' pyKMC/pyproject.toml
    grep -q '<3.14,' pyKMC/pyproject.toml || fail "Failed to bump pyproject upper Python bound (sed pattern stale?)"
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

# Rebuild mpi4py from source to match the local MPI
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
  -DPython_EXECUTABLE="$(which python)" > /dev/null 2>&1
make -j"$(sysctl -n hw.ncpu)"           > /dev/null 2>&1
make install-python                     > /dev/null 2>&1

cd "$INSTALL_DIR"

python -c "from lammps import lammps" || fail "LAMMPS Python bindings not working"
ok "LAMMPS built and installed"

# ------------------------------------------
# 5. Build IRA
# ------------------------------------------
step "Building IRA"

cd "$INSTALL_DIR/IterativeRotationsAssignments"

python -m pip install . #--quiet

cd "$INSTALL_DIR"

python -c "import ira_mod" || fail "IRA not working"
ok "IRA built and installed"

# ------------------------------------------
# 6. Build pARTn plugin
# ------------------------------------------
step "Building pARTn plugin"

cd "$INSTALL_DIR/artn-plugin"

cmake -B build \
      -DWITH_LAMMPS=ON \
      -DLAMMPS_PATH="$INSTALL_DIR/lammps/build" \
      -DARTN_INSTALL_PYTHON=ON \
      -DCMAKE_CXX_FLAGS_INIT="-std=c++17"             > /dev/null 2>&1
cmake --build build --parallel "$(sysctl -n hw.ncpu)" > /dev/null 2>&1
cmake --install build                                 > /dev/null 2>&1

cd "$INSTALL_DIR"

python -c "
import pypARTn
a=pypARTn.artn(engine='lmp')
" || fail "pypARTn not working"
ok "pARTn built and installed"

# ------------------------------------------
# 7. Verify full installation
# ------------------------------------------
step "Verifying installation"

python -c "
import ase, pykmc, ira_mod
import lammps
lmp=lammps.lammps()
import pypARTn
artn=pypARTn.artn( engine='lmp' )
lmp.command( f'plugin load {artn.lib._name}' )
print( 'Loaded libraries:' )
print( ' * liblammps   ::', lmp.lib._name )
print( ' * libartn-lmp ::', artn.lib._name )
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
# Drop any inherited pyKMC PYTHONPATH so the venv's pypARTn / ira_mod are loaded
# (not the previous install's interface modules, which can shadow site-packages)
unset PYTHONPATH
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
