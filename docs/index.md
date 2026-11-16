# pyKMC

**pyKMC** is a Python code for **adaptive kinetic Monte Carlo (aKMC)** simulations.

It currently relies on [LAMMPS](https://www.lammps.org), [pARTn](https://gitlab.com/mammasmias/artn-plugin) and [IRA](https://github.com/mammasmias/IterativeRotationsAssignments).

## Requirements

pyKMC requires Python ≥ 3.10 and three external codes:

- LAMMPS, built with the `PLUGIN`, `EXTRA-COMPUTE` and basic packages;
- pARTn, the LAMMPS plugin for saddle point searches;
- IRA, for shape matching and symmetries.

See the [installation guide](install/installation.md) for step-by-step instructions and install scripts.

## Quick start

- [User Guide](user_guide.md): how to write an input file and run a simulation.
- [Examples](https://github.com/hugomoison/pyKMC/tree/main/examples): ready-to-run systems.

## Documentation

| Section | Content |
|---|---|
| [Installation](install/installation.md) | Building LAMMPS, pARTn and IRA, and installing pyKMC |
| [User Guide](user_guide.md) | Writing an input file, section by section |
| [Parameters](parameters.md) | Every input parameter, its type and default value |
| [Developer Guide](dev_strategy_pattern.md) | Code architecture and guidelines for contributors |
| [API Reference](api/kmc.md) | Auto-generated documentation of the modules |

## License

pyKMC is distributed under the MIT license.
Source code and issues: [github.com/hugomoison/pyKMC](https://github.com/hugomoison/pyKMC).
