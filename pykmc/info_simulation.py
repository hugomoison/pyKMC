"""Module with functions to construct information dataclass for the output's files."""

from .result import (
    AtomicEnvironmentInfo,
    EventSearchOutput,
    ErrorInfo,
    Result,
    ReferenceEventSearchInfo,
    ErrorType,
    ReferenceValidEventsInfo,
    RefinementsInfo,
    EventRefinementOutput,
    EventsInfo,
    AdaptiveSearchInfo,
    ShapeConvergenceInfo,
    ShapeID,
)
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .kmc import KMC
    from .adaptive_search import ShapeSearchStats
import pandas as pd


def info_atomic_environments(
    kmc: "KMC",
    new_shapes: dict[ShapeID, list[int]],
) -> AtomicEnvironmentInfo:
    """Construct the dataclass containing information of the atomic environments.

    Parameters
    ----------
    kmc : KMC
        the kmc object.
    new_shapes : dict[ShapeID, list[int]]
        shape -> eligible atom indices needing search, at the current step.

    Returns
    -------
    AtomicEnvironmentInfo
        the dataclass containing information of the atomic environments.

    """
    atomic_environments_info = AtomicEnvironmentInfo(
        total_atomic_environments_encounter=len(kmc.visited_environments),
        n_current_atomic_environments=len(
            set(kmc.atomic_environment.atomic_environment_list)
        ),
        n_new_shapes=len(new_shapes),
    )
    if kmc.params.control.verbosity == 2:
        atom_group = {}
        for index, item in enumerate(kmc.atomic_environment.atomic_environment_list):
            if item != "crystal":
                if item not in atom_group:
                    atom_group[item] = []
                atom_group[item].append(index)
        atomic_environments_info.atoms_grouped_by_environment = list(
            atom_group.values()
        )
    return atomic_environments_info


def info_reference_event_searches(
    results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]],
) -> ReferenceEventSearchInfo:
    """Construct the dataclass containing informations on the reference event searches.

    Parameters
    ----------
    results_reference_event_searches : list[Result[EventSearchOutput, ErrorInfo]]
        list of results of the reference event searches procedure.

    Returns
    -------
    ReferenceEventSearchInfo
        the dataclass containing information on the reference event searches.

    """
    total_event_searches = len(results_reference_event_searches)
    n_success = 0
    n_fails = {
        "no_event_found": 0,
        "minima_not_matching_positions": 0,
        "minima_not_distinct": 0,
        "runtime_error": 0,
    }
    for res in results_reference_event_searches:
        if res.is_ok():
            n_success += 1
        else:
            if res.err_value().type == ErrorType.EVENT_NOT_FOUND:
                n_fails["no_event_found"] += 1
            elif res.err_value().type == ErrorType.EVENT_MINIMA_NOT_DISTINCT:
                n_fails["minima_not_distinct"] += 1
            elif res.err_value().type == ErrorType.EVENT_SEARCH_RUNTIME_ERROR:
                n_fails["runtime_error"] += 1
            else:
                n_fails["minima_not_matching_positions"] += 1
    return ReferenceEventSearchInfo(total_event_searches, n_success, n_fails)


def info_is_valid_reference_events(
    results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]],
) -> ReferenceValidEventsInfo:
    """Construct the dataclass containing informations on whether or not an event has been added to the reference table.

    Parameters
    ----------
    results_is_valid_events : list[Result[pd.DataFrame, ErrorInfo]]
        list of results of the add event procedure.

    Returns
    -------
    ReferenceValidEventsInfo
        the dataclass containing informations on whether or not an event has been added to the reference table.

    """
    n_valid_events = 0
    invalid_events = {
        "dE > emax_event": 0,
        "dE < emin_event": 0,
        "dE inverse < emin_event": 0,
        "Event asymmetric": 0,
        "Event already in reference table": 0,
        "min1 and min2 not distinct": 0,
    }
    for res in results_is_valid_events:
        if res.is_ok():
            n_valid_events += 1
        else:
            match res.err_value().type:
                case ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD:
                    invalid_events["dE > emax_event"] += 1
                case ErrorType.EVENT_ENERGY_LOWER_THAN_THRESHOLD:
                    invalid_events["dE < emin_event"] += 1
                case ErrorType.EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD:
                    invalid_events["dE inverse < emin_event"] += 1
                case ErrorType.EVENT_ASYMMETRIC:
                    invalid_events["Event asymmetric"] += 1
                case ErrorType.EVENT_NOT_NEW:
                    invalid_events["Event already in reference table"] += 1
                case ErrorType.EVENT_MINIMA_NOT_DISTINCT:
                    invalid_events["min1 and min2 not distinct"] += 1
    return ReferenceValidEventsInfo(n_valid_events, invalid_events)


def info_adaptive_search(
    shape_stats: dict[ShapeID, "ShapeSearchStats"],
) -> AdaptiveSearchInfo:
    """Construct the dataclass summarizing the adaptive event-search stopping rule.

    Parameters
    ----------
    shape_stats : dict[ShapeID, ShapeSearchStats]
        Per-shape stats returned by `KMC.adaptive_event_search`.

    Returns
    -------
    AdaptiveSearchInfo
        Summary of the adaptive event-search stopping rule for this step.

    """
    shape_infos = []
    n_converged = 0
    n_capped = 0
    n_escalated = 0
    n_skipped = 0
    for shape, st in shape_stats.items():
        if st.state == "converged":
            n_converged += 1
        elif st.state == "capped":
            n_capped += 1
        elif st.state == "skipped":
            n_skipped += 1
        if st.escalated:
            n_escalated += 1
        shape_infos.append(
            ShapeConvergenceInfo(
                shape=shape,
                n_searches=st.n_dispatched,
                n_valid_draws=st.n_valid_draws,
                n_distinct_events=len(st.rediscovery_counts),
                final_fraction=st.last_fraction,
                escalated=st.escalated,
                outcome=st.state,
            )
        )
    return AdaptiveSearchInfo(
        n_shapes=len(shape_stats),
        n_converged=n_converged,
        n_capped=n_capped,
        n_skipped=n_skipped,
        n_escalated=n_escalated,
        shapes=shape_infos,
    )


def info_refinements(
    results_refinements: list[Result[EventRefinementOutput, ErrorType]],
) -> RefinementsInfo:
    """Construct the dataclass containing information on the refinements procedure.

    Parameters
    ----------
    results_refinements : list[Result[EventRefinementOutput, ErrorType]]
        list of results from the refinement procedure.

    Returns
    -------
    RefinementsInfo
        the dataclass containing information on the refinements procedure.

    """
    n_attempts = len(results_refinements)
    n_successes = 0
    n_fails = {
        "no_match_found": {"n": 0, "ref_event": []},
        "matching_score_>_matching_threshold": {
            "n": 0,
            "ref_event": [],
            "matching_score": [],
        },
        "invalid_dE": {"n": 0, "ref_event": []},
        "invalid_minima": {"n": 0, "ref_event": []},
        "event_not_found": {
            "n": 0,
            "ref_event": [],
            "no_saddle_found": 0,
            "delr_sad_too_large": 0,
        },
        "runtime_error": {"n": 0, "ref_event": []},
    }
    for res in results_refinements:
        if res.is_ok():
            n_successes += 1
        else:
            match res.err_value().type:
                case ErrorType.PSR_NO_MATCH_FOUND:
                    n_fails["no_match_found"]["n"] += 1
                    n_fails["no_match_found"]["ref_event"].append(
                        res.err_value().variables["n_ref_event"]
                    )
                case ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD:
                    n_fails["matching_score_>_matching_threshold"]["n"] += 1
                    n_fails["matching_score_>_matching_threshold"]["ref_event"].append(
                        res.err_value().variables["n_ref_event"]
                    )
                    n_fails["matching_score_>_matching_threshold"][
                        "matching_score"
                    ].append(res.err_value().variables["matching_score"])
                case ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER:
                    n_fails["invalid_dE"]["n"] += 1
                    n_fails["invalid_dE"]["ref_event"].append(
                        res.err_value().variables["n_ref_event"]
                    )
                case ErrorType.REFINEMENT_INVALID_MINIMA:
                    n_fails["invalid_minima"]["n"] += 1
                case ErrorType.EVENT_NOT_FOUND:
                    n_fails["event_not_found"]["n"] += 1
                    n_fails["event_not_found"]["ref_event"].append(
                        res.err_value().variables["n_ref_event"]
                    )
                    for attempt in res.err_value().variables["attempts_detail"]:
                        if attempt["has_sad"]:
                            n_fails["event_not_found"]["delr_sad_too_large"] += 1
                        else:
                            n_fails["event_not_found"]["no_saddle_found"] += 1
                case ErrorType.EVENT_REFINEMENT_RUNTIME_ERROR:
                    n_fails["runtime_error"]["n"] += 1
                    n_fails["runtime_error"]["ref_event"].append(
                        res.err_value().variables["n_ref_event"]
                    )

    return RefinementsInfo(n_attempts, n_successes, n_fails)


def info_active_events(
    system_types, reference_table, active_table, idx_selected_event
) -> tuple[int, EventsInfo]:
    """Construct dataclass with active events information.

    Returns
    -------
    tuple[int, EventsInfo]
        - int: position of the selected event within the returned EventsInfo
          arrays (`idx_selected_event` is a row position in `active_table.table`,
          which the dedup/sort below reorders, so it must be remapped).
        - EventsInfo: the dataclass containing information of the active events.

    """

    # active table data
    central_atom = active_table.table["atom_index"].to_numpy(dtype=int, copy=True)
    types = np.array(system_types)[central_atom]
    reference_events = active_table.table["num_reference_event"].to_numpy(copy=True)
    dE_forward = active_table.table["dE_forward"].to_numpy(copy=True)
    k = active_table.table["k"].to_numpy(copy=True)
    refined = active_table.table["refined"].to_numpy(copy=True)

    # Needed mapping to access reference table info
    idx_ref = reference_table.table["idx_ref"].values
    mapping_id_initial = dict(zip(idx_ref, reference_table.table["id_initial"].values))
    mapping_event_id = dict(zip(idx_ref, reference_table.table["event_id"].values))
    mapping_id_saddle = dict(zip(idx_ref, reference_table.table["id_saddle"].values))
    mapping_id_final = dict(zip(idx_ref, reference_table.table["id_final"].values))
    mapping_dra = dict(zip(idx_ref, reference_table.table["dra"].values))
    mapping_backward = dict(zip(idx_ref, reference_table.table["idx_backward"].values))
    mapping_energy = dict(zip(idx_ref, reference_table.table["dE_forward"].values))

    # get info applying mapping
    initial_topologies = np.array([mapping_id_initial[ref] for ref in reference_events])
    event_ids = np.array([mapping_event_id[ref] for ref in reference_events])
    id_saddles = np.array([mapping_id_saddle[ref] for ref in reference_events])
    id_finals = np.array([mapping_id_final[ref] for ref in reference_events])
    dra_i = np.array([mapping_dra[ref] for ref in reference_events])
    backward_events = np.array([mapping_backward[ref] for ref in reference_events])
    dE_backward = np.array([mapping_energy[ref] for ref in backward_events])
    dra_f = np.array([mapping_dra[ref] for ref in backward_events])

    dE_asym = np.abs(dE_forward - dE_backward)

    # Keep only one row per (central_atom, reference_event) pair for logging, dropping symmetry-equivalent copies
    keep = (~pd.DataFrame({"c": central_atom, "r": reference_events}).duplicated(keep="first")).to_numpy()

    # Sort the kept rows by central atom
    order = np.argsort(central_atom[keep], kind="stable")

    central_atom_out = central_atom[keep][order]
    reference_events_out = reference_events[keep][order]
    idx_display = int(
        np.nonzero(
            (central_atom_out == central_atom[idx_selected_event])
            & (reference_events_out == reference_events[idx_selected_event])
        )[0][0]
    )

    return idx_display, EventsInfo(
        types=types[keep][order],
        central_atom=central_atom_out,
        initial_topologies=initial_topologies[keep][order],
        reference_events=reference_events_out,
        dE_forward=dE_forward[keep][order],
        dE_backward=dE_backward[keep][order],
        dE_asym=dE_asym[keep][order],
        k=k[keep][order],
        dra_i=dra_i[keep][order],
        dra_f=dra_f[keep][order],
        refined=refined[keep][order],
        event_id=event_ids[keep][order],
        id_initial=initial_topologies[keep][order],
        id_saddle=id_saddles[keep][order],
        id_final=id_finals[keep][order],
    )


def info_basin_events(
    system_types, reference_table, connectivity_table, exit_state
) -> tuple[int, EventsInfo]:
    """Construct dataclass with exit basin events"""

    # Only exit state
    data = connectivity_table.df[
        connectivity_table.df["transient"] == False
    ].reset_index(drop=True)
    idx_selected_event = data.index[data["state_connexion"] == exit_state][0]

    # Connectivity table data
    central_atom = data["central_atom"].to_numpy(dtype=int, copy=True)
    types = np.array(system_types)[central_atom]
    reference_events = data["event_connexion"].to_numpy(copy=True)
    dE_forward = data["dE_forward"].to_numpy(copy=True)
    k = data["k_forward"].to_numpy(copy=True)
    refined = len(central_atom) * ["T"]

    # Needed mapping to extract reference table info
    idx_ref = reference_table.table["idx_ref"].values
    mapping_id_initial = dict(zip(idx_ref, reference_table.table["id_initial"].values))
    mapping_event_id = dict(zip(idx_ref, reference_table.table["event_id"].values))
    mapping_id_saddle = dict(zip(idx_ref, reference_table.table["id_saddle"].values))
    mapping_id_final = dict(zip(idx_ref, reference_table.table["id_final"].values))
    mapping_dra = dict(zip(idx_ref, reference_table.table["dra"].values))
    mapping_backward = dict(zip(idx_ref, reference_table.table["idx_backward"].values))
    mapping_energy = dict(zip(idx_ref, reference_table.table["dE_forward"].values))

    # Apply mapping
    initial_topologies = np.array([mapping_id_initial[ref] for ref in reference_events])
    event_ids = np.array([mapping_event_id[ref] for ref in reference_events])
    id_saddles = np.array([mapping_id_saddle[ref] for ref in reference_events])
    id_finals = np.array([mapping_id_final[ref] for ref in reference_events])
    dra_i = np.array([mapping_dra[ref] for ref in reference_events])
    backward_events = np.array([mapping_backward[ref] for ref in reference_events])
    dE_backward = np.array([mapping_energy[ref] for ref in backward_events])
    dra_f = np.array([mapping_dra[ref] for ref in backward_events])

    dE_asym = np.abs(dE_forward - dE_backward)

    return idx_selected_event, EventsInfo(
        types=types,
        central_atom=central_atom,
        initial_topologies=initial_topologies,
        reference_events=reference_events,
        dE_forward=dE_forward,
        dE_backward=dE_backward,
        dE_asym=dE_asym,
        k=k,
        dra_i=dra_i,
        dra_f=dra_f,
        refined=refined,
        event_id=event_ids,
        id_initial=initial_topologies,
        id_saddle=id_saddles,
        id_final=id_finals,
    )
