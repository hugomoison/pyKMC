#!/bin/bash
#
# pyKMC Linux Installation Script
# Tested for: Ubuntu/Debian, RHEL/CentOS/Rocky;
#             Alliance/DRAC clusters (validated on Trillium login + compute, 2026-08-31)
#
# Usage:
#   chmod +x install_pykmc_linux.sh
#   ./install_pykmc_linux.sh
#
# Override Python interpreter:
#   PYTHON_BIN=/usr/bin/python3.12 ./install_pykmc_linux.sh
#
# This script will create a "pykmc_install" directory in the current location
# and install everything inside it.
#
# Pin pARTn / IRA to other revisions (default: the SHAs validated on 2026-07-28):
#   ARTN_REF=main IRA_REF=main ./install_pykmc_linux.sh
#
# Install a different pyKMC branch (default: develop):
#   PYKMC_BRANCH=main ./install_pykmc_linux.sh
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

# Alliance/DRAC HPC clusters (Trillium, Narval, Beluga, ...): no sudo — software
# comes from Lmod modules, which must be loaded BEFORE running this script.
ON_ALLIANCE=false
if [ -n "${CC_CLUSTER:-}" ] || [ -d /cvmfs/soft.computecanada.ca ]; then
    ON_ALLIANCE=true
    echo "Alliance/DRAC cluster detected (${CC_CLUSTER:-cvmfs present}); skipping system package installs."
    echo "Make sure the toolchain modules are loaded (check exact versions with 'module spider <name>'):"
    echo "  module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0"
    echo "NOTE: mpi4py must be >= 4.0.2 for pyKMC (mpi4py/4.0.0 is too old)."
    echo "Run this script on a LOGIN node (compute nodes have no internet access),"
    echo "from a directory the compute nodes can write to (on Trillium: \$SCRATCH —"
    echo "\$HOME and \$PROJECT are read-only from compute jobs)."
elif command -v apt >/dev/null 2>&1; then
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
elif command -v dnf > /dev/null 2>&1; then
    PKGS=""
    rpm -q gcc-c++       >/dev/null 2>&1 || PKGS="$PKGS gcc-c++"
    rpm -q gcc-gfortran  >/dev/null 2>&1 || PKGS="$PKGS gcc-gfortran"
    rpm -q cmake         >/dev/null 2>&1 || PKGS="$PKGS cmake"
    rpm -q openmpi-devel >/dev/null 2>&1 || PKGS="$PKGS openmpi-devel"
    rpm -q fftw-devel    >/dev/null 2>&1 || PKGS="$PKGS fftw-devel"
    rpm -q lapack-devel  >/dev/null 2>&1 || PKGS="$PKGS lapack-devel"
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

# An inherited PYTHONPATH (e.g. a developer shell exporting another checkout's
# artn-plugin / IterativeRotationsAssignments interface dirs) shadows everything this
# script installs into its venv, so the verification steps would test the WRONG copies.
# Alliance clusters: keep the module system's entries (they carry the EBPYTHONPREFIXES
# site customization that makes module-provided mpi4py visible) and strip only pyKMC
# interface dirs. Everywhere else: drop it outright for the duration of the install.
if [ -n "${PYTHONPATH:-}" ]; then
    if $ON_ALLIANCE; then
        PYTHONPATH=$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -vE 'artn-plugin/interface|IterativeRotationsAssignments/interface' | paste -sd: -)
        if [ -n "$PYTHONPATH" ]; then export PYTHONPATH; else unset PYTHONPATH; fi
    else
        echo "Dropping inherited PYTHONPATH for the duration of the install: $PYTHONPATH"
        unset PYTHONPATH
    fi
fi

# Verify compilers are available
command -v gfortran >/dev/null 2>&1 || fail "gfortran not found"
command -v mpicc    >/dev/null 2>&1 || fail "mpicc not found"
command -v mpicxx   >/dev/null 2>&1 || fail "mpicxx not found"
command -v mpif90   >/dev/null 2>&1 || fail "mpif90 not found"
command -v cmake    >/dev/null 2>&1 || fail "cmake not found"

# LAMMPS + pARTn initialize MPI in "singleton" mode when verified with bare `python`; on
# OpenMPI 5 systems (seen on Ubuntu 26.04) that can crash hard enough to take the whole
# terminal with it, while the same code under `mpirun -np 1` gets a proper PMIx
# environment and works. Alliance login nodes: keep bare python (validated on Trillium;
# mpirun policy on login nodes varies by cluster).
if ! $ON_ALLIANCE && command -v mpirun >/dev/null 2>&1; then
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
# on Debian/Ubuntu every python besides the default needs its own pythonX.Y-venv package,
# and uv-managed pythons reached through a ~/.local/bin symlink build venvs whose python
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
    fail "No Python 3.10-3.13 able to create a virtualenv found. Install one with its
       pythonX.Y-venv package. Distros already shipping a newer default (Ubuntu 26.04
       ships 3.14) need a parallel 3.13, e.g. from the deadsnakes PPA:
         sudo add-apt-repository ppa:deadsnakes/ppa
         sudo apt install python3.13 python3.13-venv python3.13-dev
       — or pass PYTHON_BIN=<path> explicitly to try an untested newer version."
fi

venv_works "$PYTHON_BIN" \
    || fail "$PYTHON_BIN cannot create a working virtualenv. On Debian/Ubuntu install its pythonX.Y-venv package; for a uv-managed python pass the real interpreter path as PYTHON_BIN, not the ~/.local/bin shim."

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    fail "pyKMC needs Python >= 3.10, found $PYTHON_VERSION at $PYTHON_BIN"
elif [ "$PYTHON_MINOR" -gt 13 ]; then
    # Only reachable via an explicit PYTHON_BIN (auto-selection stops at 3.13).
    echo "WARNING: Python $PYTHON_VERSION is newer than the tested range (3.10-3.13)."
    echo "Proceeding because PYTHON_BIN was set explicitly. Some pinned dependencies may"
    echo "not ship wheels for it yet (a from-source pydantic-core build needs Rust)."
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

# On Alliance clusters use virtualenv --no-download (the recommended pattern;
# the venv then sees module-provided packages such as mpi4py via EBPYTHONPREFIXES).
if $ON_ALLIANCE && command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download ./pykmc_env
else
    "$PYTHON_BIN" -m venv ./pykmc_env
fi
source pykmc_env/bin/activate

python -m pip install --upgrade pip --quiet
python -m pip install -e ./pyKMC --quiet
ok "pyKMC installed"

# mpi4py: on Alliance clusters the module (loaded before venv creation) provides an
# ABI-matched mpi4py — do NOT pip-install over it. Elsewhere, rebuild from source
# to match the system MPI.
if $ON_ALLIANCE; then
    python -c "import mpi4py" >/dev/null 2>&1 \
        || fail "mpi4py module not loaded. Load it BEFORE running this script (e.g. 'module load mpi4py/4.1.0') — never pip-install mpi4py on Alliance clusters."
    python -c "import mpi4py, sys; v = tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in mpi4py.__version__.split('.')[:3]); sys.exit(0 if v >= (4, 0, 2) else 1)" \
        || fail "mpi4py module too old ($(python -c 'import mpi4py; print(mpi4py.__version__)')) — pyKMC needs >= 4.0.2 (e.g. 'module load mpi4py/4.1.0')."
    ok "mpi4py provided by cluster module ($(python -c 'import mpi4py; print(mpi4py.__version__)'))"
else
    export CC=mpicc
    export CXX=mpicxx
    export FC=mpif90
    python -m pip install --no-binary mpi4py mpi4py --force-reinstall --quiet
    ok "mpi4py rebuilt from source"
fi

# ------------------------------------------
# 3. Build LAMMPS
# ------------------------------------------
step "Building LAMMPS (this may take a few minutes)"

cd "$INSTALL_DIR/lammps"
mkdir -p build && cd build

# Cap build parallelism on shared cluster login nodes (override with MAKE_JOBS=N)
if $ON_ALLIANCE; then
    MAKE_JOBS="${MAKE_JOBS:-8}"
else
    MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
fi

# Build output goes to log files (tail -f them to watch progress).
# PKG_PHONON provides the dynamical_matrix command needed by HTST rate prefactors.
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
# install-python needs internet (it pip-installs `build` into an isolated env) — on
# clusters run this script on a login node.
python -m pip uninstall -y lammps >/dev/null 2>&1 || true
make install-python                     > make_install.log 2>&1 \
    || fail "LAMMPS install-python failed — see $(pwd)/make_install.log (on clusters this step needs a login node: it downloads build tools from PyPI)"

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

# Instantiate rather than just `import ira_mod`. At the pinned revision the bare import happens to
# dlopen the library anyway (ira_mod.py runs `version, _ = IRA().get_version()` at module level),
# but that is an incidental side effect to lean a build check on, and a bare `import ira_mod` is a
# silent pass whenever a directory named ira_mod shadows the package as an empty namespace package.
# Instantiating states the intent, and prints the version into the install log as provenance.
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
cmake --install build                     > artn_install.log 2>&1 \
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
ACTIVATE

if $ON_ALLIANCE; then
    cat >> "$INSTALL_DIR/activate.sh" << 'ACTIVATE'
# Strip stale pyKMC interface dirs from PYTHONPATH (they shadow the venv's
# pypARTn / ira_mod) but KEEP everything else — on Alliance clusters the python
# module's PYTHONPATH entry carries the site customization that processes
# EBPYTHONPREFIXES (how module-provided packages like mpi4py become visible),
# so a blanket unset silently breaks mpi4py.
if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -vE 'artn-plugin/interface|IterativeRotationsAssignments/interface' | paste -sd: -)
    if [ -n "$PYTHONPATH" ]; then export PYTHONPATH; else unset PYTHONPATH; fi
fi
echo "pyKMC environment activated (Alliance/DRAC cluster)."
echo "If modules are not loaded yet (e.g. in a fresh job shell), load them first:"
echo "  module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0"
echo "Run pyKMC inside a Slurm job with srun (NOT mpirun):"
echo '  srun --ntasks=$SLURM_NTASKS --distribution=block:block --cpu-bind=cores --mem-bind=local python -m pykmc -in input.in'
ACTIVATE
else
    cat >> "$INSTALL_DIR/activate.sh" << 'ACTIVATE'
# Drop any inherited pyKMC PYTHONPATH so the venv's pypARTn / ira_mod are loaded
# (not a previous install's interface modules, which can shadow site-packages)
unset PYTHONPATH
echo "pyKMC environment activated. Run with:"
echo "  mpirun -n 8 python -m pykmc -in input.in"
ACTIVATE
fi
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
if $ON_ALLIANCE; then
    echo '  srun --ntasks=$SLURM_NTASKS --distribution=block:block --cpu-bind=cores --mem-bind=local python -m pykmc -in input.in'
else
    echo "  mpirun -n 8 python -m pykmc -in input.in"
fi
echo ""
