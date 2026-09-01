import numpy as np
import ctypes
from ...otfml import (
    OTFML_MAX_FLAG,
    OTFML_TOL_FLAG,
    OTFML_LATCH,
    session_dump_path,
    otf_thermo_path,
)


def initialize_parameters(engine):
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary p p p")
    engine.command("atom_modify map array")  #! necessary for scatter atoms
    engine.command("atom_modify sort 0 0.0")  #! necessary for partn


def initialize_potential(engine, params):
    pair_style = params.lammps.pair_style
    pair_coeff = params.lammps.pair_coeff
    engine.command("pair_style {}".format(pair_style))
    engine.command("pair_coeff {}".format(pair_coeff))
    for cmd in params.lammps.setup_commands or []:
        engine.command(cmd)

    if pair_style.strip().startswith("mtp/extrapolation"):
        try:
            engine.command(
                "fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1"
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "Please use the MLIP-3 style extrapolation for configuration mode MTPs"
                in msg
            ):
                raise RuntimeError(
                    "The loaded MTP is in configuration mode. "
                    "Current pyKMC OTFML expects neighborhood-mode `mtp/extrapolation` "
                    "with per-atom `f_extrapolation_grade` support. "
                    "Use a neighborhood-mode MTP or disable OTFML"
                ) from exc
            raise

    if params.control.otfml:
        if not pair_style.strip().startswith("mtp/extrapolation"):
            raise RuntimeError("OTFML requires `pair_style mtp/extrapolation`.")
        gamma_tol = params.otfml.gamma_tolerance
        gamma_max = params.otfml.gamma_max
        dump_path = session_dump_path(engine.engine_id).as_posix()
        engine.command(f"variable {OTFML_TOL_FLAG} internal 0")
        engine.command(f"variable {OTFML_MAX_FLAG} internal 0")
        engine.command(f"compute max_grade all reduce max f_extrapolation_grade")
        engine.command(f"variable max_grade equal c_max_grade")
        engine.command(f'variable dump_skip equal "v_max_grade < {gamma_tol:.4f}"')
        engine.command(
            f"dump extrapolative_structures_dump all custom 1 {dump_path} id type x y z f_extrapolation_grade"
        )
        engine.command(f"dump_modify extrapolative_structures_dump append yes")
        engine.command(f"dump_modify extrapolative_structures_dump skip v_dump_skip")
        engine.command(
            f"fix extreme_extrapolation all halt 5 v_max_grade > {gamma_max:.4f} error continue"
        )
        _setup_otf_latch(engine, gamma_tol, gamma_max)
        engine.command(f"log {otf_thermo_path(engine).as_posix()}")
        # engine.command("echo none")
        engine.command(f"thermo 1")
        engine.command(
            f"thermo_style custom step pe v_max_grade v_{OTFML_LATCH} v_{OTFML_TOL_FLAG} v_{OTFML_MAX_FLAG}"
        )
        engine.command("thermo_modify line yaml flush no")


def _setup_otf_latch(engine, gamma_tol: float, gamma_max: float) -> None:
    """Register a python-style variable that latches the OTF flag internal variables.

    Evaluated every minimization step via thermo_style, so flags are updated
    the moment grade crosses either threshold — not just at the final step.
    """
    latch_code = (
        f"def _latch_otf_flags(handle, max_grade):\n"
        f"    from lammps import lammps\n"
        f"    lmp = lammps(ptr=handle)\n"
        f"    if max_grade >= {gamma_tol:.4f}: lmp.set_internal_variable('{OTFML_TOL_FLAG}', 1.0)\n"
        f"    if max_grade >= {gamma_max:.4f}: lmp.set_internal_variable('{OTFML_MAX_FLAG}', 1.0)\n"
        f"    return lmp.extract_variable('{OTFML_TOL_FLAG}')\n"
    )
    engine.command(
        f"python _latch_otf_flags"
        f" input 2 SELF v_max_grade"
        f" return v_{OTFML_LATCH}"
        f" format pff"
        f' here """{latch_code}"""'
    )
    engine.command(f"variable {OTFML_LATCH} python _latch_otf_flags")


def set_positions(engine, positions):
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)


def get_potential_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    engine.command("run 0")
    result = engine.lmp.get_thermo("pe")
    return result
