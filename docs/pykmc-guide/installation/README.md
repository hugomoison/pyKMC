# Install pyKMC and OVITO

Choose one computing environment: [macOS](#a-macos--apple-silicon), [Windows with WSL2](#b-windows--wsl2--ubuntu), or [Trillium](#c-trillium). Install [OVITO](#d-ovito-and-completion) on your own computer. If pyKMC already works, go straight to your platform's activation and verification steps.

pyKMC runs kinetic Monte Carlo, LAMMPS supplies energies and forces, pARTn searches transitions, and IRA matches structures. MPI connects the parallel processes. The installer builds these components together; see the [glossary](../glossary.md) as needed.

pyKMC comes from the upstream [hugomoison/pyKMC repository](https://github.com/hugomoison/pyKMC). The installation scripts are from the [Kerrator fork at a fixed revision](https://github.com/Kerrator/pyKMC/tree/599f162e63ae8bbecf84c48d0a22b8053dae4fe7/docs/install), which supplies needed installation fixes and the directory layout below. The [maintenance record](../maintenance/validation.md#installer-source-selection) explains that choice.

Commands assume the guide is at `~/pykmc-guide`. Keep your existing location if different and substitute its path, using Tab completion if convenient. Shell path variables are optional ([path help](../paths.md)). Start in a terminal without another Python/Conda environment active; keep the installation log and any patches.

<a id="a-local-mac--apple-silicon"></a>
<a id="macos"></a>

## A. macOS — Apple Silicon

### A1. Check prerequisites

In your local terminal, check Apple's developer tools, Git and [Homebrew](https://brew.sh/):

```bash
xcode-select -p
git --version
brew --version
```

If missing, install the developer tools with `xcode-select --install` and Homebrew using its official instructions. Then install/check Python:

```bash
brew install python@3.13
"$(brew --prefix python@3.13)/bin/python3.13" --version
```

Use a native Apple Silicon toolchain; do not mix it with Intel/Rosetta packages. This route uses Python 3.13.

### A2. Download and build

The directory below is a suggested location. Choose a fresh parent directory if it already contains an unsuccessful `pykmc_install/`.

```bash
mkdir -p ~/software/pyKMC_setup
cd ~/software/pyKMC_setup
curl -fL -o install_pykmc_mac.sh \
  https://raw.githubusercontent.com/Kerrator/pyKMC/599f162e63ae8bbecf84c48d0a22b8053dae4fe7/docs/install/install_pykmc_mac.sh
less install_pykmc_mac.sh
```

Inspect the script and press q to leave the viewer. Stop if the download failed. Then run:

```bash
set -o pipefail
PYTHON_BIN="$(brew --prefix python@3.13)/bin/python3.13" MAKE_JOBS=4 PYKMC_BRANCH=develop \
  bash install_pykmc_mac.sh 2>&1 | tee install.log
echo "Installer exit status: $?"
```

The installer obtains missing Homebrew dependencies, creates `pykmc_install/` and builds the libraries. `MAKE_JOBS=4` limits compilation concurrency; `tee` saves output, and `pipefail` preserves failure status.

Expect final verification and exit status 0. If it stops at `pypARTn not working` with `artn_create: symbol not found`, preserve the partial installation and follow the [version-specific recovery appendix](macos-partn-recovery.md). Do not apply that patch to unrelated errors or rerun the entire installer blindly.

### A3. Activate and verify

After successful installation or recovery:

```bash
source ~/software/pyKMC_setup/pykmc_install/pykmc_env/bin/activate
unset PYTHONPATH
command -v python
```

Adjust the path if you installed elsewhere. Repeat activation in each new terminal; use `deactivate` when finished. Keep your normal zsh shell; use `bash` to execute the supplied installer, preparation and launch scripts. Directly sourcing the virtual environment avoids the top-level activation script's Bash-specific path handling. Clearing `PYTHONPATH` prevents inherited overrides from selecting another installation.

Run the [library checker](scripts/check_libraries.py), substituting your guide's actual path if needed:


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -n 1 python ~/pykmc-guide/installation/scripts/check_libraries.py
```

After the library checker prints `PASS`, test communication separately (no atomic simulation is run):

```bash
mpirun -n 2 python ~/pykmc-guide/installation/scripts/check_mpi.py
```

Expect `PASS: 2 MPI ranks; allgather, allreduce and broadcast`, then continue to [OVITO and completion](#d-ovito-and-completion). This checks MPI communication, not a complete eight-rank pyKMC calculation.

<a id="d-windows--wsl2--ubuntu"></a>
<a id="windows"></a>

## B. Windows — WSL2 + Ubuntu

WSL2 provides a Linux environment on Windows. Use Ubuntu for pyKMC and Windows OVITO for visualization. The WSL installation and complete simulation workflow still need an end-to-end test.

### B1. Set up Ubuntu

Follow [Microsoft's WSL installation instructions](https://learn.microsoft.com/en-us/windows/wsl/install). In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu
```

Restart if requested, open Ubuntu, and create its Linux user account. In PowerShell, `wsl -l -v` should show Ubuntu using version 2. If already installed, check the existing setup.

### B2. Choose Linux locations

Keep the guide, installation and runs in the Ubuntu home directory, following [Microsoft's filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems). Use the Ubuntu terminal for the remaining computing steps.

### B3. Install pyKMC in Ubuntu

Build pyKMC inside Ubuntu with the [Linux installer](https://github.com/Kerrator/pyKMC/blob/599f162e63ae8bbecf84c48d0a22b8053dae4fe7/docs/install/install_pykmc_linux.sh), after checking Ubuntu/Python compatibility through the [pyKMC support channel](../README.md#support-and-references). This is an installation entry point, rather than a tested Ubuntu build sequence. The macOS Homebrew commands and Trillium module stack are platform-specific.

### B4. Activate and verify in Ubuntu

After a successful Linux installation, activate it and check the executables. Substitute your actual installation path:

```bash
source ~/software/pyKMC_setup/pykmc_install/activate.sh &&
command -v python mpirun
```

Python and MPI should resolve to the intended Linux tools. Run the same [library and two-process MPI checks](#a3-activate-and-verify) using the guide's Ubuntu path.

### B5. Run an example

Use the local blocks in each example, with Linux paths such as `~/pykmc-runs` rather than `C:\...` or `/Users/...`. The supplied launchers need 8 MPI CPU slots; a 4-core/8-thread machine may expose only 4 slots. Check the available MPI layout before launching, rather than forcing oversubscription.

### B6. Visualize locally

Install Windows OVITO Basic using the instructions below. In an Ubuntu result directory, `explorer.exe .` opens that location in File Explorer. Open the trajectory and reference there, or copy both to a Windows analysis folder.

Keep logs if a build/check fails and use the [support channels](../README.md#support-and-references). The macOS recovery patch has a different, specific purpose.

<a id="b-trillium"></a>
<a id="trillium"></a>

## C. Trillium

<a id="b1-copy-the-guide-and-connect"></a>

### C1. Copy the guide and connect

Follow [SciNet's access instructions](https://scinethpc.ca/getting-started/): obtain an Alliance account, request Trillium access, enable MFA, and upload your SSH public key to CCDB. You also need an authorized allocation for jobs. Consult [Trillium's documentation](https://docs.alliancecan.ca/wiki/Trillium) for current policies.

First copy or later update: use the same procedure below from your local terminal. Edit the guide path and username. The trailing slash copies the guide's contents, avoiding an extra nested folder; private `_runs/` folders are excluded. Keep a separate backup of remote edits before updating. The transfer replaces matching guide files, keeps their previous versions as `.previous`, and does not delete other remote files. Repeated updates are not a versioned backup.

```bash
rsync -av --dry-run --backup --suffix=.previous \
  --exclude='_runs/' --exclude='.DS_Store' --exclude='.git/' \
  ~/pykmc-guide/ YOUR_USERNAME@trillium.alliancecan.ca:pykmc-guide/
```

Review the preview, then perform the copy:

```bash
rsync -av --backup --suffix=.previous \
  --exclude='_runs/' --exclude='.DS_Store' --exclude='.git/' \
  ~/pykmc-guide/ YOUR_USERNAME@trillium.alliancecan.ca:pykmc-guide/
ssh YOUR_USERNAME@trillium.alliancecan.ca
```

Complete key authentication and MFA when prompted. The remaining Trillium commands run in that SSH terminal. Use this transfer procedure for downloaded guide folders; `git pull` applies only to a Git checkout.

<a id="b2-load-modules-and-choose-the-installation-directory"></a>

### C2. Load modules and choose the installation directory

```bash
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0
module list
command -v python mpicc mpicxx mpif90 gfortran cmake
python -c "import mpi4py; print(mpi4py.__version__)"
cd "${SCRATCH:?SCRATCH is not set}"
mkdir -p pyKMC_install
cd pyKMC_install
```

Use this compatible stack, including Python 3.12.4 and mpi4py 4.1.0; do not substitute the Mac's Python 3.13. Modules provide the underlying compiler/MPI stack, while the virtual environment holds project packages. If a module is unavailable, inspect `module spider MODULE_NAME` and ask for help before changing versions.

The installer creates an inner `pykmc_install/`. If it already works, skip to C4. Scratch is working storage, including the installation placed there, not a permanent software archive. Keep installer/source revisions, patches and important results in backed-up storage. Check the current site policy for retention and cleanup rules. If a long-lived installation is needed, agree a suitable location with support before building; moving an existing venv can break its paths.

<a id="b3-download-and-build-on-the-login-node"></a>

### C3. Download and build on the login node

Use this cluster-aware installer, not a script that attempts system installation with `sudo`. Compilation needs internet access; use modest parallelism and follow current login-node policy. Simulations belong in Slurm jobs.

```bash
curl -fL -o install_pykmc_linux.sh \
  https://raw.githubusercontent.com/Kerrator/pyKMC/599f162e63ae8bbecf84c48d0a22b8053dae4fe7/docs/install/install_pykmc_linux.sh
less install_pykmc_linux.sh
```

Inspect, press q, and stop if the download failed. Then run:

```bash
set -o pipefail
PYTHON_BIN="$(command -v python3.12)" MAKE_JOBS=4 PYKMC_BRANCH=develop \
  bash install_pykmc_linux.sh 2>&1 | tee install.log
echo "Installer exit status: $?"
```

Expect final verification and exit status 0. If it fails, keep the partial build and logs; do not submit a calculation.

<a id="b4-activate-and-verify"></a>

### C4. Activate and verify

Load the C2 module stack before activation, including in each fresh login or job shell:

```bash
source "$SCRATCH/pyKMC_install/pykmc_install/activate.sh"
command -v python
python ~/pykmc-guide/installation/scripts/check_libraries.py
```

Use your actual installation path if different. Expect Python inside `pykmc_env/bin/` and a final `PASS`. This is a short single-process library test; run the later parallel simulation with the supplied Slurm template, not directly on the login node.

Unlike the Mac route, do not blanket-unset `PYTHONPATH` here: the Alliance Python/module configuration uses it to expose module-provided packages such as mpi4py. The cluster-aware activation script preserves the required entries.

<details>
<summary>Optional: test two-rank MPI communication before the first example</summary>

After loading modules and activating, request a short allocation from a directory under scratch. Replace `YOUR_ALLOCATION`:

```bash
cd "$SCRATCH"
salloc --account=YOUR_ALLOCATION --partition=compute --nodes=1 --ntasks=2 --cpus-per-task=1 --time=00:05:00
```

Once allocation is granted, in its shell run:

```bash
srun --ntasks=2 --kill-on-bad-exit=1 python ~/pykmc-guide/installation/scripts/check_mpi.py
exit
```

Use the actual guide location if different. Expect `PASS: 2 MPI ranks; allgather, allreduce and broadcast`. `exit` releases the allocation shell. This optional communication test may allocate a whole node. Its Trillium execution remains untested; see the validation records.

</details>

<a id="c-ovito-and-completion"></a>
<a id="ovito"></a>

## D. OVITO and completion

Install OVITO Basic on your own computer from the [official website](https://www.ovito.org/), following its [installation instructions](https://www.ovito.org/manual/installation.html). Basic supports the trajectory and Wigner–Seitz visualization used here, including results downloaded from Trillium.

The library checker prints interpreter/library paths, checks IRA and `eam/alloy`, and loads pARTn into LAMMPS. Expect `PASS: imports, IRA, LAMMPS eam/alloy and pARTn plugin loading`. This confirms library integration; the separate two-rank check tests communication, and the example run tests the complete workflow.

Keep `install.log`, the installer and any patch. Save Python version/package records in your installation parent directory with `python --version` and `python -m pip freeze`; Trillium users should also save `module list`. Keep the entire working `pykmc_install/` at its original path: editable source checkouts and compiled libraries remain part of the environment. Do not copy environments between machines or upload them to the guide repository.

Once activation, the library checks and OVITO work, continue to [Ni monovacancy](../examples/ni-monovacancy/README.md).

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `sudo` or `mpicc` errors on Trillium | Use the pinned cluster-aware installer and load the C2 modules first. |
| Missing Python package or wrong interpreter | Reactivate the matching environment and inspect `command -v python`; do not replace module-provided mpi4py with pip as a first fix. |
| macOS `artn_create` missing | Use the [conditional recovery appendix](macos-partn-recovery.md) only for its specified revision/error. |
| `activate` file not found | The virtual-environment path ends in `pykmc_env/bin/activate`, not `pykmc_env/activate`. |
| Existing installation directory | Reuse a working installation or preserve the failed attempt and choose a fresh parent directory. |
| Build appears silent or fails | Inspect `install.log`, `lammps/build/{cmake_config,make,make_install}.log` and `artn-plugin/artn_{cmake,build,install}.log` under the installation root. |

For other failures, use the [support channels](../README.md#support-and-references), including the command, error, OS and software versions. Remove credentials and personal paths before sharing logs. Do not force a patch or continue after a failed check.

Platform tests and source versions are recorded in [validation](../maintenance/validation.md). A fixed installer revision does not freeze the moving `develop` branch or all dependencies.
