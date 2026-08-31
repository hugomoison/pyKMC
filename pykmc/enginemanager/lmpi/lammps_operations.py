import numpy as np
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes
import pypARTn
from ...utils.io_utils import capture_output
from ...utils.geometry import compute_distances
from ...activevolume.active_volume import (
    reset,
    redefine_atoms,
    partn_search_AV,
    partn_refine_AV,
    position_results_AV,
)
from ...atomic_environment import AtomicEnvironment
from ...system import Configuration
from ...otfml import (
    OTFML_MAX_FLAG,
    OTFML_TOL_FLAG,
    read_otf_thermo,
    OTFExtrapolationFlags,
)
from .engine_primitives import (
    initialize_parameters,
    initialize_potential,
    set_positions,
    get_potential_energy,
)

from ...result import (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)


def initialize_system(engine, configuration: Configuration, params=None):

    # system parameters
    natoms = len(configuration.types)
    cell = configuration.cell
    types = configuration.types
    x = configuration.positions.flatten()  # Lammps format

    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

    ind = np.linspace(0, natoms - 1, natoms).astype(int)
    ind += 1  # Lammps id start at 1

    type_order = (
        params.lammps.type_order
        if (params and params.lammps.type_order)
        else list(dict.fromkeys(types))
    )
    if set(type_order) != set(types):
        raise ValueError(
            f"type_order {type_order} does not match the elements in the system {sorted(set(types))}"
        )
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(type_order)
    }
    types = [map_type[element]["ref"] for element in types]  # map to integer

    # lammps create system
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    engine.lmp.create_atoms(natoms, ind, types, x)
    # Set masses
    for key in map_type.keys():
        engine.command("mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"]))
    # Label atoms name to type :
    engine.command(
        "labelmap atom "
        + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
    )
    engine.type_labels = {v["ref"]: key for key, v in map_type.items()}


def setup_otf_cycle(engine, params):
    """Load a new pair_style and reset OTFML state for the next retrain cycle.

    Reissuing fix pair with the same ID/style replaces the old FixPair instance,
    resetting its lasttime state and rebinding it to the current pair style while
    preserving active dump/compute references by ID.
    """
    engine.command(f"pair_style {params.lammps.pair_style}")
    engine.command(f"pair_coeff {params.lammps.pair_coeff}")
    engine.command(
        "fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1"
    )
    reset_otf_flags(engine)


def reset_otf_flags(engine) -> None:
    """Clear the latched OTF extrapolation flags."""
    engine.lmp.set_internal_variable(OTFML_TOL_FLAG, 0.0)
    engine.lmp.set_internal_variable(OTFML_MAX_FLAG, 0.0)


def get_thermo_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read OTF flags from the already-parsed engine.otf_thermo block."""
    # Read last thermo YAML block for diagnostics only.
    engine.command("log none")  # force flush
    otf_thermo = read_otf_thermo(engine)

    if otf_thermo is None:
        raise RuntimeError("OTF thermo data not available on engine.")

    tol_col = f"v_{OTFML_TOL_FLAG}"
    max_col = f"v_{OTFML_MAX_FLAG}"

    return OTFExtrapolationFlags(
        extrapolated=bool(max(otf_thermo[tol_col]) > 0),
        extreme_extrapolated=bool(max(otf_thermo[max_col]) > 0),
    )


def get_lammps_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read the current latched OTF extrapolation flags from the engine."""

    def extract_scalar(name: str) -> float:
        # try:
        value = engine.lmp.extract_variable(name, None, 0)
        # except TypeError:
        #     value = engine.lmp.extract_variable(name)
        return float(value)

    return OTFExtrapolationFlags(
        extrapolated=bool(extract_scalar(OTFML_TOL_FLAG)),
        extreme_extrapolated=bool(extract_scalar(OTFML_MAX_FLAG)),
    )


def get_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read the current latched OTF extrapolation flags from the engine."""

    return get_lammps_otf_flags(engine)
    # return get_thermo_otf_flags(engine)


def _build_extrapolation_error(
    flags: OTFExtrapolationFlags,
    *,
    phase: str,
    message: str,
    variables: dict,
):
    if flags.extreme_extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTREME_EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    if flags.extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    return None


def minimize(engine, params, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    engine.command("min_style {}".format(params.lammps.min_style))
    engine.command("minimize {}".format(params.lammps.minimize))


def get_total_energy(engine, positions=None):
    result = get_potential_energy(engine, positions=positions)
    if engine.rank == 0:
        return result


def get_positions(engine) -> np.ndarray:
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        result = np.ctypeslib.as_array(result)
        result = np.reshape(result, (-1, 3))
    else:
        result = None
    return engine.engine_comm.bcast(result, root=0)


def get_types(engine) -> list[str]:
    int_types = engine.lmp.gather_atoms("type", 0, 1)
    if engine.rank == 0:
        types = [engine.type_labels[t] for t in int_types]
    else:
        types = None
    return engine.engine_comm.bcast(types, root=0)


def get_cell(engine) -> np.ndarray:
    boxlo, boxhi, xy, yz, xz, periodicity, box_change = engine.lmp.extract_box()
    return np.diag(np.array(boxhi) - np.array(boxlo))


def set_cell(engine, cell) -> None:
    engine.command(
        "change_box all x final 0.0 {} y final 0.0 {} z final 0.0 {} units box".format(
            cell[0][0], cell[1, 1], cell[2, 2]
        )
    )


def set_types(engine, types) -> None:
    label_to_ref = {label: ref for ref, label in engine.type_labels.items()}
    int_types = [label_to_ref[t] for t in types]
    c_array = (ctypes.c_int * len(int_types))(*int_types)
    engine.lmp.scatter_atoms("type", 0, 1, c_array)


def get_configuration(engine) -> Configuration:
    return Configuration(
        types=get_types(engine), positions=get_positions(engine), cell=get_cell(engine)
    )


def set_configuration(engine, configuration: Configuration) -> None:
    set_cell(engine, configuration.cell)
    set_types(engine, configuration.types)
    set_positions(engine, positions=configuration.positions)


def minimize_with_results(engine, params, configuration: Configuration):
    """
    Minimize and return the minimized Configuration and the total energy.
    """
    set_configuration(engine, configuration)
    atoms_frozen = _make_frozen_group(engine, params, configuration)
    _apply_frozen_fix(engine, "f_frozen_min", atoms_frozen)
    minimize(engine, params)
    _remove_frozen_fix(engine, "f_frozen_min", atoms_frozen)
    _delete_frozen_group(engine, atoms_frozen)
    new_configuration = get_configuration(engine)
    total_energy = get_total_energy(engine)
    if engine.rank == 0:
        return new_configuration, total_energy


def minimize_freeze_core(engine, params, core_idx):
    """
    Freeze directly translated atoms and minimize to relax surrounding atoms
    """

    if core_idx is not None:
        core_ids = [idx + 1 for idx in core_idx]
        engine.command(f"group frozen_group id {' '.join(map(str, core_ids))}")
        engine.command("fix freeze frozen_group setforce 0.0 0.0 0.0")
        engine.command(f"min_style {params.lammps.min_style}")
        engine.command(f"minimize {params.lammps.frz_min}")
        engine.command("unfix freeze")
        engine.command("group frozen_group delete")


def _make_frozen_group(engine, params, configuration: Configuration) -> bool:
    """Resolve frozen atoms and create g_frozen group. Returns True if any atoms are frozen."""
    if params.frozen_atoms is None:
        return False
    frozen_ae = AtomicEnvironment(
        style="region",
        region=params.frozen_atoms,
        configuration=configuration,
    )
    frozen_indices = frozen_ae.get_atoms_with_id("in")
    if not frozen_indices:
        return False
    lammps_ids = " ".join(str(i + 1) for i in frozen_indices)  # 1-based LAMMPS IDs
    engine.command(f"group g_frozen id {lammps_ids}")
    return True


def _apply_frozen_fix(engine, fix_name: str, atoms_frozen: bool) -> None:
    """Add a setforce 0 0 0 fix on g_frozen under the given name."""
    if atoms_frozen:
        engine.command(f"fix {fix_name} g_frozen setforce 0.0 0.0 0.0")


def _remove_frozen_fix(engine, fix_name: str, atoms_frozen: bool) -> None:
    """Remove a setforce fix previously added by _apply_frozen_fix."""
    if atoms_frozen:
        engine.command(f"unfix {fix_name}")


def _delete_frozen_group(engine, atoms_frozen: bool) -> None:
    """Delete the g_frozen group after all fixes on it have been removed."""
    if atoms_frozen:
        engine.command("group g_frozen delete")


def _reset_engine_state(engine, params, configuration: Configuration) -> None:
    """Rebuild the local LAMMPS state after a search crashes mid-command."""
    engine.command("clear")
    initialize_parameters(engine)
    initialize_system(engine, configuration.copy(), params)
    initialize_potential(engine, params)


@capture_output()
def partn_search(engine, params, central_atom_idx: int, configuration: Configuration):
    cell = configuration.cell
    types = configuration.types
    try:
        if params.control.active_volume == True:
            atom_map, central_lammps_id = partn_search_AV(
                engine, params, central_atom_idx, configuration
            )
        else:
            atom_map = None
            central_lammps_id = [central_atom_idx + 1]
            set_configuration(engine, configuration)

        if params.control.otfml:
            reset_otf_flags(engine)

        artn = pypARTn.artn(engine="lammps")
        engine.command(f"plugin load {artn.lib._name}")
        atoms_frozen = _make_frozen_group(engine, params, configuration)
        _apply_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
        engine.command("fix 10 all artn dmax {}".format(params.partn.dmax))
        _apply_frozen_fix(engine, "f_frozen_post", atoms_frozen)
        engine.command("min_style fire")

        artn.reset_input()
        artn.set("filout", "artn.out." + str(engine.engine_id))
        artn.set("engine_units", "lammps/metal")
        artn.set("verbose", params.partn.verbosity)
        artn.set("struc_format_out", "none")
        artn.set("delr_thr", params.partn.delr_thr)

        artn.set("lpush_final", True)
        artn.set("lmove_nextmin", False)
        artn.set("zseed", params.partn.zseed)

        artn.set("push_mode", params.partn.push_mode)
        if params.partn.push_mode == "rad":
            artn.set("push_dist_thr", params.partn.push_dist_thr)
        artn.set("push_step_size", params.partn.push_step_size)
        artn.set("push_ids", central_lammps_id)
        artn.set("ninit", params.partn.ninit)

        artn.set("lanczos_min_size", params.partn.lanczos_min_size)
        artn.set("lanczos_max_size", params.partn.lanczos_max_size)
        artn.set("lanczos_disp", params.partn.lanczos_disp)
        artn.set("lanczos_eval_conv_thr", params.partn.lanczos_eval_conv_thr)

        artn.set("eigval_thr", params.partn.eigval_thr)
        artn.set("eigen_step_size", params.partn.eigen_step_size)
        artn.set("nsmooth", params.partn.nsmooth)
        artn.set("neigen", params.partn.neigen)
        artn.set("alpha_mix_cr", params.partn.alpha_mix_cr)
        artn.set("nnewchance", params.partn.nnewchance)

        if params.partn.nperp is not None:
            artn.set("nperp", params.partn.nperp)
        if params.partn.nperp_limitation is not None:
            artn.set("nperp_limitation", np.array(params.partn.nperp_limitation))
        else:
            artn.set("lnperp_limitation", False)

        artn.set("forc_thr", params.partn.forc_thr)
        artn.set("push_over", params.partn.push_over)
        engine.command(f"minimize 1e-6 1e-8 10000 {params.partn.nevalf_max}")
    except RuntimeError as exc:
        recovery_error = None
        try:
            _reset_engine_state(engine, params, configuration)
        except Exception as recovery_exc:
            recovery_error = recovery_exc

        details = str(exc)
        if recovery_error is not None:
            details = (
                f"{details}; recovery failed with "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_SEARCH_RUNTIME_ERROR,
                message="Runtime error during event search.",
                details=details,
                variables={"central_atom_index": central_atom_idx},
            )
        )

    engine.command("unfix 10")
    _remove_frozen_fix(engine, "f_frozen_post", atoms_frozen)
    _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
    _delete_frozen_group(engine, atoms_frozen)

    if engine.rank == 0:
        if params.control.otfml:
            extrapolation_error = _build_extrapolation_error(
                get_otf_flags(engine),
                phase="search",
                message="Search extrapolated and must be retried.",
                variables={
                    "central_atom_index": central_atom_idx,
                },
            )
            if extrapolation_error is not None:
                return extrapolation_error

        err = artn.get_error()
        has_extract = (
            artn.extract("has_sad")
            and artn.extract("has_min1")
            and artn.extract("has_min2")
        )
        if err[0] == 0 and has_extract:
            E_sad = artn.extract("etot_sad")
            E_min1 = artn.extract("etot_min1")
            E_min2 = artn.extract("etot_min2")

            if params.control.active_volume == True:
                min1_configuration, min2_configuration, saddle_configuration, index_move = (
                    position_results_AV(params, artn, atom_map, configuration)
                )
            else:
                min1_configuration = Configuration(types=types, positions=artn.extract("tau_min1"), cell=cell)
                min2_configuration = Configuration(types=types, positions=artn.extract("tau_min2"), cell=cell)
                saddle_configuration = Configuration(types=types, positions=artn.extract("tau_sad"), cell=cell)

                # find atom that moves the most (PBC-aware, so an atom crossing
                # the periodic boundary between min1 and saddle isn't mistaken
                # for a small mover)
                dist = compute_distances(min1_configuration, saddle_configuration)
                index_move = np.argmax(dist)

            # Whether min1/min2 are genuinely distinct (EVENT_MINIMA_NOT_DISTINCT)
            # and whether either matches the live configuration this search
            # was launched from (previously EVENT_MINIMA_NOT_MATCH_POSITIONS)
            # are no longer decided here: both used a raw, whole-cell,
            # unaligned `compute_delr_max`, which can false-reject on
            # incidental far-field drift or a benign permutation during the
            # full-cell minimization. Distinctness is now checked via local
            # shape-match once cataloguing computes id_min1/id_min2 anyway
            # (`ReferenceEventTable.is_valid_new_event`); "does it match what
            # we searched from" is classified by `EventSearch` afterwards,
            # which has the graph/IRA machinery and the atom's live id this
            # module doesn't. A search that reaches this point with a
            # coherent saddle+2-minima result is always returned `Ok` --
            # cataloguing and connectivity classification decide the rest.
            delr1 = artn.extract("delr_min1")
            delr2 = artn.extract("delr_min2")

            dE_forward = E_sad - E_min1
            dE_backward = E_sad - E_min2

            if delr1 <= delr2:
                return Ok(
                    EventSearchOutput(
                        central_atom_index=central_atom_idx,
                        dE_forward=dE_forward,
                        dE_backward=dE_backward,
                        min1=min1_configuration,
                        saddle=saddle_configuration,
                        min2=min2_configuration,
                        move_atom_index=index_move,
                    )
                )

            return Ok(
                EventSearchOutput(
                    central_atom_index=central_atom_idx,
                    dE_forward=dE_backward,
                    dE_backward=dE_forward,
                    min1=min2_configuration,
                    saddle=saddle_configuration,
                    min2=min1_configuration,
                    move_atom_index=index_move,
                )
            )
        else:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND,
                    message="No event found",
                    details=err,
                )
            )


@capture_output()
def partn_refine(
    engine,
    params,
    central_atom_idx: int,
    configuration: Configuration,
    saddle_idx=None,
    saddle_positions=None,
    minimize_outter_atoms: bool = True,
    num_reference_event: int | None = None,
    symmetry_index: int | None = None,
):
    positions = configuration.positions
    cell = configuration.cell
    types = configuration.types
    try:
        if params.control.active_volume == True:
            E_init, atom_map, central_lammps_id = partn_refine_AV(
                engine,
                params,
                central_atom_idx,
                configuration,
                saddle_idx,
                saddle_positions,
            )
        else:
            central_lammps_id = [central_atom_idx + 1]
            atom_map = None
            set_configuration(engine, configuration)
            E_init = get_potential_energy(engine)
            # saddle_positions is only given when configuration is the base
            # (unmoved) system and the caller wants this delta merged in
            # (see RefinementPreparer). A caller that already moved
            # configuration to the saddle itself (BasinsGenericEvents'
            # non-active-volume refine_absorbing) leaves it None -- the
            # configuration set above is already the guess.
            if saddle_positions is not None:
                guess = configuration.copy()
                guess[saddle_idx] = saddle_positions
                set_configuration(engine, guess)
            if minimize_outter_atoms:
                minimize_freeze_core(engine, params, saddle_idx)

        if params.control.otfml:
            reset_otf_flags(engine)

        artn = pypARTn.artn(engine="lammps")
        engine.command(f"plugin load {artn.lib._name}")
        artn.reset_input()
        artn.set("filout", "artn.out." + str(engine.engine_id))
        artn.set("engine_units", "lammps/metal")
        artn.set("verbose", params.partn.verbosity)
        artn.set("struc_format_out", "none")
        artn.set("delr_thr", params.partn.delr_thr)

        artn.set("lpush_final", False)
        artn.set("lmove_nextmin", False)
        artn.set("zseed", params.partn.zseed)

        artn.set("push_mode", params.partn.r_push_mode)
        if params.partn.push_mode == "rad":
            artn.set("push_dist_thr", params.partn.r_push_dist_thr)
        artn.set("push_step_size", params.partn.r_push_step_size)
        artn.set("push_ids", central_lammps_id)  # fortran start at 1
        artn.set("ninit", params.partn.r_ninit)

        artn.set("lanczos_min_size", params.partn.r_lanczos_min_size)
        artn.set("lanczos_max_size", params.partn.r_lanczos_max_size)
        artn.set("lanczos_disp", params.partn.r_lanczos_disp)
        artn.set("lanczos_eval_conv_thr", params.partn.r_lanczos_eval_conv_thr)

        artn.set("eigval_thr", params.partn.r_eigval_thr)
        artn.set("eigen_step_size", params.partn.r_eigen_step_size)
        artn.set("nsmooth", params.partn.r_nsmooth)
        artn.set("neigen", params.partn.r_neigen)
        artn.set("alpha_mix_cr", params.partn.r_alpha_mix_cr)
        artn.set("nnewchance", params.partn.r_nnewchance)

        if params.partn.r_nperp is not None:
            artn.set("nperp", params.partn.r_nperp)
        if params.partn.r_nperp_limitation is not None:
            artn.set("nperp_limitation", np.array(params.partn.r_nperp_limitation))
        else:
            artn.set("lnperp_limitation", False)

        artn.set("forc_thr", params.partn.r_forc_thr)

        max_attempts = params.partn.r_max_attempts
        inner_attempt = 0
        attempts_detail = []
        atoms_frozen = _make_frozen_group(engine, params, configuration)
        _apply_frozen_fix(engine, "f_frozen_pre", atoms_frozen)

        while inner_attempt < max_attempts:
            exit_flag = False
            result = None
            engine.command("fix 10 all artn dmax {}".format(params.partn.r_dmax))
            _apply_frozen_fix(engine, "f_frozen_post", atoms_frozen)
            engine.command("min_style fire")
            engine.command(f"minimize 1e-6 1e-8 10000 {params.partn.r_nevalf_max}")
            engine.command("unfix 10")
            _remove_frozen_fix(engine, "f_frozen_post", atoms_frozen)

            if engine.rank == 0:
                if params.control.otfml:
                    extrapolation_error = _build_extrapolation_error(
                        get_otf_flags(engine),
                        phase="refine",
                        message="Refinement extrapolated and must be retried.",
                        variables={
                            "central_atom_index": central_atom_idx,
                            "num_reference_event": num_reference_event,
                            "symmetry_index": symmetry_index,
                        },
                    )
                    if extrapolation_error is not None:
                        exit_flag = True
                        result = extrapolation_error
                if not exit_flag:
                    err = artn.get_error()
                    has_extract = artn.extract("has_sad")

                    if err[0] == 0 and has_extract:
                        delr_sad = artn.extract("delr_sad")
                        if delr_sad < params.partn.r_delr_sad_thr:
                            E_sad = artn.extract("etot_sad")
                            E_result = E_sad - E_init
                            saddlepositions = artn.extract("tau_sad")

                            if params.control.active_volume == True:
                                saddlepositions_results = positions.copy()
                                for i, atom_idx in enumerate(atom_map):
                                    saddlepositions_results[atom_idx][0] = (
                                        saddlepositions[i][0]
                                    )
                                    saddlepositions_results[atom_idx][1] = (
                                        saddlepositions[i][1]
                                    )
                                    saddlepositions_results[atom_idx][2] = (
                                        saddlepositions[i][2]
                                    )
                            else:
                                saddlepositions_results = saddlepositions

                            exit_flag = True
                            result = Ok(
                                EventRefinementOutput(
                                    central_atom_index=central_atom_idx,
                                    saddle=Configuration(
                                        types=types,
                                        positions=saddlepositions_results,
                                        cell=cell,
                                    ),
                                    E_saddle=E_result,
                                    num_reference_event=num_reference_event,
                                    symmetry_index=symmetry_index,
                                    refined="T",
                                )
                            )
                        else:
                            attempts_detail.append(
                                {
                                    "attempt": inner_attempt,
                                    "err": err,
                                    "has_sad": True,
                                    "delr_sad": delr_sad,
                                }
                            )
                    else:
                        attempts_detail.append(
                            {
                                "attempt": inner_attempt,
                                "err": err,
                                "has_sad": bool(has_extract),
                                "delr_sad": None,
                            }
                        )
            exit_flag = engine.local_engine_comm.bcast(exit_flag, root=0)
            if exit_flag:
                _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
                _delete_frozen_group(engine, atoms_frozen)
                return result

            inner_attempt += 1
            artn.set("zseed", params.partn.zseed)

        else:
            _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
            _delete_frozen_group(engine, atoms_frozen)
            if engine.rank == 0:
                err = artn.get_error()
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_NOT_FOUND,
                        message="no event found",
                        details=err,
                        variables={"attempts_detail": attempts_detail},
                    )
                )
            return None
    except RuntimeError as exc:
        recovery_error = None
        try:
            _reset_engine_state(engine, params, configuration)
        except Exception as recovery_exc:
            recovery_error = recovery_exc

        details = str(exc)
        if recovery_error is not None:
            details = (
                f"{details}; recovery failed with "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_REFINEMENT_RUNTIME_ERROR,
                message="Runtime error during event refinement.",
                details=details,
                variables={
                    "central_atom_index": central_atom_idx,
                    "num_reference_event": num_reference_event,
                    "symmetry_index": symmetry_index,
                },
            )
        )
