"""Shared input checks for the guide's eight-rank, fresh-run recipes.

Preparation copies this file into the run directory for both local and Slurm
launchers. Read the installed pyKMC configuration model, not a second set of
parser defaults. This helper does not edit inputs or start a KMC calculation.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import sys


# These are the existing chapter checklists, not new physical parameter choices.
RECIPES = {
    "ni-monovacancy": ("eam/alloy", "Ni_v6_2.0_LKBeland2016.eam", ["Ni"], False, False),
    "ni-fcc-sia": ("eam/alloy", "Ni_v6_2.0_LKBeland2016.eam", ["Ni"], False, False),
    "ni-monovacancy-sia": ("eam/alloy", "Ni_v6_2.0_LKBeland2016.eam", ["Ni"], False, True),
    "si-monovacancy": ("sw", "Si.sw", ["Si"], False, False),
    "cu-vacancy-sia": ("eam/alloy", "Cu01.eam.alloy", ["Cu"], False, False),
    "fe-bivacancy": ("eam/alloy", "FeC-Becquart2013.eam", ["Fe"], True, False),
    "nicr-slab": ("eam/alloy", "Bonny_2013_NiFeCr.eam", ["Cr", "Ni"], True, False),
}


def load_config_class():
    """Load the installed standalone model without pyKMC's MPI-heavy __init__.

    The reviewed revisions have a self-contained config.py. If that layout/API
    changes, fail visibly; never fall back to guessed defaults or another copy.
    """
    package = importlib.util.find_spec("pykmc")
    if package is None or not package.submodule_search_locations:
        raise ValueError("pyKMC is not installed in this Python environment")
    paths = [Path(root) / "config.py" for root in package.submodule_search_locations]
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise ValueError("Cannot locate one installed pyKMC config.py; review guide/source compatibility")
    name = "_guide_installed_pykmc_config"
    spec = importlib.util.spec_from_file_location(name, paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclasses and type annotations.
    spec.loader.exec_module(module)
    return module.Config


def validate(config, example):
    pair_style, potential, species, basin, active_volume = RECIPES[example]
    control = config.control
    if (control.engine != "lammps" or control.n_steps < 1
            or control.n_sessions != 7 or control.engine_use_rank_0
            or control.group_size != -1):
        raise ValueError("Use engine = lammps, positive n_steps, n_sessions = 7, "
                         "engine_use_rank_0 = False and group_size = -1 for the eight-rank recipe")
    # Relative names must remain valid after Slurm copies inputs to run-JOBID/.
    if Path(control.initial_config) != Path("initial_config.xyz"):
        raise ValueError("Set initial_config = ./initial_config.xyz")
    for filename in ("initial_config.xyz", potential):
        if not Path(filename).is_file() or Path(filename).stat().st_size == 0:
            raise ValueError(f"Missing or empty input: {filename}")
    coeff = shlex.split(config.lammps.pair_coeff)
    if (config.lammps.pair_style.strip() != pair_style or len(coeff) != 3 + len(species)
            or coeff[:2] != ["*", "*"] or coeff[3:] != species
            or Path(coeff[2]) != Path(potential)):
        raise ValueError(f"Use pair_style = {pair_style} and pair_coeff = * * ./{potential} {' '.join(species)}")
    if control.basin != basin or control.active_volume != active_volume:
        raise ValueError(f"This chapter uses basin = {basin} and active_volume = {active_volume}")
    if any(getattr(control, field) for field in
           ("recycle", "bias", "restart_file", "reference_table", "visited_environments")):
        raise ValueError("Use the fresh baseline: no recycling, bias or imported state")
    for field, filename in (("trajectory_output", "trajkmc.xyz"),
                            ("reference_table_output", "reference_table.pickle"),
                            ("visited_environments_output", "visited_environments.pickle")):
        if Path(getattr(control, field)) != Path(filename):
            raise ValueError(f"Keep {field} at its default local path for this recipe")
    if example == "fe-bivacancy" and (config.atomicenvironment.style != "cna/graph"
                                      or config.atomicenvironment.rnei != 3.2):
        raise ValueError("Review the Fe recipe: cna/graph and rnei = 3.2")
    if example == "nicr-slab" and config.atomicenvironment.atom_coloring_mode not in ("full", "grey"):
        raise ValueError("Choose full or grey for this comparison")
    return pair_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, choices=RECIPES)
    parser.add_argument("--check-lammps", action="store_true",
                        help="Also check local venv and load LAMMPS; no minimization or dynamics")
    args = parser.parse_args()
    try:
        if args.check_lammps and (not os.environ.get("VIRTUAL_ENV")
                or Path(sys.prefix).resolve() != Path(os.environ["VIRTUAL_ENV"]).resolve()):
            raise ValueError("python is not using the activated virtual environment")
        Config = load_config_class()
        pair_style = validate(Config.from_ini_file("input.in"), args.example)
        if args.check_lammps:
            from lammps import lammps
            engine = lammps(cmdargs=["-log", "none", "-screen", "none"])
            try:
                if not engine.has_style("pair", pair_style):
                    raise ValueError(f"LAMMPS lacks pair_style {pair_style}; check the MANYBODY package installation")
            finally:
                engine.close()
    except Exception as exc:
        parser.exit(2, f"FAIL: preflight; no simulation started: {exc}\n")
    print(f"PASS: {args.example} input layout (installed pyKMC configuration model)")
    if args.check_lammps:
        print(f"PASS: LAMMPS pair_style {pair_style} is available")


if __name__ == "__main__":
    main()
