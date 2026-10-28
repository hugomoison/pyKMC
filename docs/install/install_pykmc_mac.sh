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
# This script will create a "pykmc_install" directory in the current location
# and install everything inside it.
#
# Install a different pyKMC branch (default: develop):
#   PYKMC_BRANCH=main ./install_pykmc_mac.sh
#
# Pin pARTn / IRA to other revisions (default: the SHAs validated on 2026-07-28):
#   ARTN_REF=main IRA_REF=main ./install_pykmc_mac.sh
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

# An inherited PYTHONPATH (e.g. a developer shell exporting another checkout's
# artn-plugin / IterativeRotationsAssignments interface dirs) shadows everything this
# script installs into its venv, so the verification steps would test the WRONG copies.
if [ -n "${PYTHONPATH:-}" ]; then
    echo "Dropping inherited PYTHONPATH for the duration of the install: $PYTHONPATH"
    unset PYTHONPATH
fi

# Verify compilers are available
command -v gfortran >/dev/null 2>&1 || fail "gfortran not found. Install with: brew install gcc"
command -v mpicc    >/dev/null 2>&1 || fail "mpicc not found. Install with: brew install open-mpi"
command -v mpicxx   >/dev/null 2>&1 || fail "mpicxx not found. Install with: brew install open-mpi"
command -v mpif90   >/dev/null 2>&1 || fail "mpif90 not found. Install with: brew install open-mpi"

# LAMMPS + pARTn initialize MPI in "singleton" mode when verified with bare `python`; on
# some OpenMPI 5 systems (seen on Ubuntu 26.04) that can crash hard enough to take the
# whole terminal with it, while the same code under `mpirun -np 1` gets a proper PMIx
# environment and works. Use mpirun for the MPI-touching verification steps.
if command -v mpirun >/dev/null 2>&1; then
    PYRUN="mpirun -np 1 python"
else
    PYRUN="python"
fi

ok "All prerequisites found"

# ------------------------------------------
# 0b. Select a supported Python interpreter
# ------------------------------------------
step "Selecting Python interpreter"

# A candidate must be able to create a WORKING venv, not just report the right version:
# uv-managed pythons reached through a ~/.local/bin symlink build venvs whose python
# cannot locate its own stdlib (ModuleNotFoundError: No module named 'encodings').
venv_works() {
    local probe
    probe=$(mktemp -d) || return 1
    if "$1" -m venv --without-pip "$probe/v" >/dev/null 2>&1 \
        && "$probe/v/bin/python" -c "import encodings, ensurepip" >/dev/null 2>&1; then
        rm -rf "$probe"
        return 0
    fi
    rm -rf "$probe"
    return 1
}

find_supported_python() {
    for py in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then
            local ver
            ver=$("$py" -c "import sys; print(sys.version_info.minor)")
            if [ "$ver" -ge 10 ] && [ "$ver" -le 13 ] && venv_works "$py"; then
                echo "$py"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(find_supported_python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    echo "No Python 3.10-3.13 able to create a virtualenv found. Installing python@3.13 with Homebrew..."
    brew install python@3.13
    PYTHON_BIN="$(brew --prefix python@3.13)/bin/python3.13"
fi

venv_works "$PYTHON_BIN" \
    || fail "$PYTHON_BIN cannot create a working virtualenv. For a uv-managed python pass the real interpreter path as PYTHON_BIN, not the ~/.local/bin shim."

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 13 ]; then
    fail "Supported Python range is 3.10-3.13, found $PYTHON_VERSION at $PYTHON_BIN"
fi

ok "Using Python $PYTHON_VERSION at $(command -v "$PYTHON_BIN")"

# ------------------------------------------
# 1. Create working directory and clone repos
# ------------------------------------------
step "Cloning repositories"

# Deliberately NOT named "pykmc": a directory of that name makes `import pykmc` resolve to it as an
# empty namespace package whenever python runs from this directory's PARENT. The import succeeds,
# __file__ is None, and the real failure surfaces later and misleadingly as
#   ImportError: cannot import name 'NeighborsList' from 'pykmc' (unknown location)
INSTALL_DIR="$(pwd)/pykmc_install"

if [ -d "$INSTALL_DIR" ]; then
    fail "Directory $INSTALL_DIR already exists. Remove it or run from a different location."
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# pARTn and IRA are pinned by SHA to the revisions validated on 2026-07-28. Neither has a usable
# release tag: artn-plugin's only tag (v1.0.0) predates the LAMMPS plugin, and IRA's IRA_v2.2.0 tag
# sits behind the 2.2.0 code. Unpinned, two runs of this script weeks apart install different
# software — between 2026-06-10 and 2026-07-28, pARTn moved 8224be16 -> edea36ac and IRA
# 7f011ba -> 3cb0c29. Override to track upstream HEAD deliberately: ARTN_REF=main IRA_REF=main.
IRA_REF="${IRA_REF:-3cb0c299e2e664f8131948b90f9926869d42459c}"
ARTN_REF="${ARTN_REF:-edea36aca8215a1d484b3b8695ecb9676fe56498}"

# The pyKMC branch this script installs. Keep the default in sync with the branch the
# script itself ships on (develop here, main on main), so downloading the script from a
# branch installs that same branch. Override with PYKMC_BRANCH=<branch>.
PYKMC_BRANCH="${PYKMC_BRANCH:-develop}"
echo "Installing pyKMC branch: $PYKMC_BRANCH"

git clone -b "$PYKMC_BRANCH" https://github.com/hugomoison/pyKMC.git
git clone -b stable_22Jul2025_update3 --depth 1 https://github.com/lammps/lammps.git
git clone https://github.com/mammasmias/IterativeRotationsAssignments.git
git clone https://gitlab.com/mammasmias/artn-plugin.git

git -C IterativeRotationsAssignments checkout --quiet "$IRA_REF" \
    || fail "Could not check out IRA revision '$IRA_REF'"
git -C artn-plugin checkout --quiet "$ARTN_REF" \
    || fail "Could not check out pARTn revision '$ARTN_REF'"

# Record what actually landed, so a simulation result can be attributed to a stack later.
echo "Resolved source revisions:"
for repo in pyKMC lammps IterativeRotationsAssignments artn-plugin; do
    printf '  %-32s %s\n' "$repo" "$(git -C "$repo" rev-parse HEAD)"
done

ok "All repositories cloned"

# ------------------------------------------
# 2. Create virtual environment and install pyKMC
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
# 3. Build LAMMPS
# ------------------------------------------
step "Building LAMMPS (this may take a few minutes)"

cd "$INSTALL_DIR/lammps"
mkdir build && cd build

# Override build parallelism with MAKE_JOBS=N
MAKE_JOBS="${MAKE_JOBS:-$(sysctl -n hw.ncpu)}"

# PKG_PHONON provides the dynamical_matrix command needed by HTST rate prefactors.
# Without it LAMMPS raises nothing and every HTST event silently degrades to k0.
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
  -DPython_EXECUTABLE="$(which python)" > cmake_config.log 2>&1 \
    || fail "LAMMPS cmake configure failed — see $(pwd)/cmake_config.log"
make -j"$MAKE_JOBS"                     > make.log 2>&1 \
    || fail "LAMMPS build failed — see $(pwd)/make.log"

# Drop any lammps wheel pip pulled from PyPI (pyKMC's pyproject lists `lammps` as a
# dependency) so the wheel built from THIS LAMMPS is the one in the venv.
python -m pip uninstall -y lammps >/dev/null 2>&1 || true
make install-python                     > make_install.log 2>&1 \
    || fail "LAMMPS install-python failed — see $(pwd)/make_install.log"

cd "$INSTALL_DIR"

python -c "from lammps import lammps" || fail "LAMMPS Python bindings not working"
ok "LAMMPS built and installed"

# ------------------------------------------
# 4. Build IRA
# ------------------------------------------
step "Building IRA"

cd "$INSTALL_DIR/IterativeRotationsAssignments"

# IRA builds via scikit-build-core, which does propagate a failed cmake build as a non-zero pip
# exit code (the full CMake/compiler output is printed even under --quiet). It is not built
# out-of-tree in a way that leaves a log behind, so there is no log file to point at — say so.
python -m pip install . --quiet \
    || fail "IRA build failed — the CMake/compiler output above is the full log (no log file is written)"

cd "$INSTALL_DIR"

# Instantiate rather than just `import ira_mod`: a bare import is a silent pass whenever a
# directory named ira_mod shadows the package, and instantiating dlopens the shared library,
# printing the version into the install log as provenance.
python -c "
import ira_mod
print('IRA library OK:', ira_mod.IRA().get_version())
" || fail "IRA shared library not loadable (libira.so missing, or unresolved symbols)"
ok "IRA built and installed"

# ------------------------------------------
# 5. Build pARTn plugin
# ------------------------------------------
step "Building pARTn plugin"

cd "$INSTALL_DIR/artn-plugin"

cmake -B build \
      -DWITH_LAMMPS=ON \
      -DLAMMPS_PATH="$INSTALL_DIR/lammps/build" \
      -DARTN_INSTALL_PYTHON=ON \
      -DCMAKE_CXX_FLAGS_INIT="-std=c++17" > artn_cmake.log 2>&1 \
    || fail "pARTn cmake configure failed — see $(pwd)/artn_cmake.log"
cmake --build build --parallel "$MAKE_JOBS" > artn_build.log 2>&1 \
    || fail "pARTn build failed — see $(pwd)/artn_build.log"
cmake --install build                       > artn_install.log 2>&1 \
    || fail "pARTn install failed — see $(pwd)/artn_install.log"

cd "$INSTALL_DIR"

$PYRUN -c "
import pypARTn
a=pypARTn.artn(engine='lmp')
" || fail "pypARTn not working"
ok "pARTn built and installed"

# ------------------------------------------
# 6. Verify full installation
# ------------------------------------------
step "Verifying installation"

$PYRUN -c "
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
# 7. Create activation script
# ------------------------------------------
cat > "$INSTALL_DIR/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Source this file to activate the pyKMC environment:
#   source /path/to/pykmc_install/activate.sh

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
