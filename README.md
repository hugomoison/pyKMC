# pyKMC

Python framework for adaptive Kinetic Monte Carlo (aKMC) simulations.  
Full documentation: [LINK](https://hugomoison.github.io/pyKMC)

## Requirements

pyKMC currently requires LAMMPS, pARTn and IRA (no alternative implementations exist yet).

- Python ≥ 3.10
- MPI and [mpi4py](https://mpi4py.readthedocs.io) (linked against the MPI you run with)
- [LAMMPS](https://github.com/lammps/lammps) : built with cmake as a shared library, with `PKG_PLUGIN`, `PKG_EXTRA-COMPUTE` and the pair-style packages your potentials need (e.g. `MANYBODY` for EAM)
- [pARTn](https://gitlab.com/mammasmias/artn-plugin/-/tree/v1.1.0) `v1.1.0` : LAMMPS plugin for saddle point searches
- [IRA](https://github.com/mammasmias/IterativeRotationsAssignments) : shape matching library

## Installation

The full guide (using an existing LAMMPS, mpi4py, clusters, troubleshooting) is in [docs/install/installation.md](docs/install/installation.md).

### Automated install

One-shot scripts build the whole stack (pyKMC + LAMMPS + pARTn + IRA) into a `pykmc/` folder under the current directory: [`install_pykmc_linux.sh`](docs/install/install_pykmc_linux.sh), [`install_pykmc_mac.sh`](docs/install/install_pykmc_mac.sh).

### Manual install

1. **pyKMC**, in a dedicated virtual environment:

   ```bash
   python3 -m venv /path/to/pykmc_env
   source /path/to/pykmc_env/bin/activate
   cd /path/to/pyKMC
   pip install -e .
   ```

2. **LAMMPS** (tested with `stable_22Jul2025_update3`), built with cmake (the make build no longer supports the PLUGIN package), then installed into the venv:

   ```bash
   cd /path/to/lammps && mkdir build && cd build
   cmake ../cmake \
     -DBUILD_SHARED_LIBS=on -DLAMMPS_EXCEPTIONS=on \
     -DPKG_KSPACE=on -DPKG_MANYBODY=on -DPKG_MOLECULE=on -DPKG_RIGID=on \
     -DPKG_EXTRA-COMPUTE=on -DPKG_PLUGIN=on -DPKG_PHONON=on \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 \
     -DPython_EXECUTABLE=$(which python)
   make -j$(nproc)
   make install-python
   ```

3. **pARTn** `v1.1.0` (later versions may introduce breaking changes), which installs the `pypARTn` module into the venv:

   ```bash
   git clone -b v1.1.0 --depth 1 https://gitlab.com/mammasmias/artn-plugin.git
   cd artn-plugin
   cmake -B build -DWITH_LAMMPS=ON -DLAMMPS_PATH=/path/to/lammps/build -DARTN_INSTALL_PYTHON=ON
   cmake --build build && cmake --install build
   ```

4. **IRA**:

   ```bash
   cd /path/to/ira
   pip install .
   ```

Check the installation:

```bash
python -c "from lammps import lammps; import ase, pykmc, ira_mod, pypARTn; print('All imports OK')"
```

## Usage

Running pyKMC requires MPI and an input file. Use at least `n_sessions + 1` ranks (rank 0 runs the KMC loop, the others drive the LAMMPS sessions). See the [documentation](https://hugomoison.github.io/pyKMC) and [`examples/`](examples/) for the input file format and examples.

```bash
mpirun -n 8 python -m pykmc -in input.in
```

## Package structure

```
pyKMC/
├── pykmc/          Source package (entry point: python -m pykmc); modules are described in the API reference
├── examples/       Ready-to-run systems (Ni, Cu, Fe, NiCr, cSi…) with input templates and expected outputs
├── tests/          pytest suite and its test data
├── benchmarks/     Benchmark scripts for individual components
├── docs/           Documentation sources (MkDocs); install guides and scripts in docs/install/
├── scripts/        Developer scripts (parameters doc generation, validation)
├── utils/          Analysis, visualisation and initial-configuration generators
├── pyproject.toml  Package metadata and dependencies
└── mkdocs.yml      Documentation site configuration
```
