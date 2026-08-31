from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import (
    System,
    Parameters,
    Configuration,
    NeighborsList,
    AtomicEnvironment,
    ReferenceEventTable,
    PointSetRegistration,
    check_match,
    Reconstruction,
)
from typing import Optional
from ..utils import geometry
from ..rate_constant import compute_rate_Eyring
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, Err, ErrorInfo, ErrorType, BasinOutput, ShapeID

# TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
# TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
# TODO: Think about parallized exploration
# TODO: Could think of refining transient -> absorbing event when exploring
# TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
# TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.


@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList]
    transient: bool = False
    visited: bool = False

    def release_heavy_objects(self) -> None:
        """Release heavy objects"""
        self.neighbors_list = None
        self.environment = None

    def ensure_full_state(self, params: Parameters) -> None:
        if self.system is not None:
            if self.neighbors_list is None:
                self.neighbors_list = NeighborsList(
                    self.system,
                    params.atomicenvironment.rnei,
                    params.atomicenvironment.rcut,
                    params.atomicenvironment.rnei_pairs,
                )
            if self.environment is None:
                self.environment = AtomicEnvironment(
                    params.atomicenvironment.style,
                    self.neighbors_list.neighbors_list["rnei"],
                    self.neighbors_list.neighbors_list["rcut"],
                    params.atomicenvironment.neighbors_add,
                    coordination_threshold=params.atomicenvironment.coordination_threshold,
                    types=self.system.types,
                    coloring_mode=params.atomicenvironment.coloring_mode,
                )


class BasinsGenericEvents:
    def __init__(
        self, params: Parameters, reference_table, known_environments, manager
    ) -> None:
        self.params = params  # Parameters object with basins parameters
        self.explorer = None  # object to explore a state in the basin
        self.reference_table = reference_table  # Object with reference generic events
        self.manager = manager  # object to do external task (minimize, refine)

        self.connectivity_table = None  # Dataframe of basin connexion state
        self.selected_event = None  # The selected event after basin exploration
        self.current_state = None  # Current state where we're at
        self.states_to_explore = None  # List of state to explore
        self.explored_states = None  # List of state that we already explored
        self.states: dict[int, StateData] = {}  # Dictionnary of StateDate
        self.known_environments = known_environments
        self.absorbing_saddle_configurations: dict[tuple[int, int], Configuration] = {}

    def detection(self, params) -> bool:
        """Utility method."""
        return self.detector.detection(**params)

    def execute(self, system):
        """
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        # initialize the basin
        self._initialize(system)

        # Every state discovered *during* exploration gets this check inside
        # construct_connexion_table()'s loop before being explored -- but
        # state 0 is pre-added by _initialize() and never passes through
        # that branch, so it has to be checked here instead. Without this,
        # a state 0 with an uncatalogued atom silently explores to zero
        # connectivity entries, and reorder_states_index() then returns an
        # empty mapping that doesn't cover state 0's own index.
        self.states[0].ensure_full_state(self.params)
        is_unknown, _ = self.is_states_has_unknown_environments(self.states[0])
        if is_unknown:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_UNKNOWN_INITIAL_ENVIRONMENT,
                    message="Basin: the state basin exploration was entered on has an atom whose shape is not catalogued.",
                )
            )

        # explore the basin
        result = self.construct_connexion_table()
        if not result.is_ok():
            return result
        # reorder states index
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}
        # Refine absorbing states
        self.manager.use_local()
        result = self.refine_absorbing(system)
        if not result.is_ok():
            return result
        # apply selector algorithm to find t_exit and exit_state
        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok():
            return result
        # Construct output KMC needs
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state

        from_state, event_idx, central_atom, sym_idx, is_transient = (
            self.connectivity_table.get_transition_to_state(target_state=exit_state)
        )
        # Ensure from_state is state are full
        self.states[from_state].ensure_full_state(self.params)

        neighbors = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )
        saddle_configuration = self.absorbing_saddle_configurations[(from_state, exit_state)]
        return Ok(
            BasinOutput(
                initial_system_configuration=self.states[from_state].system.configuration,
                central_atom=central_atom,
                saddle_configuration=saddle_configuration,
                final_configuration=self.states[exit_state].system.configuration[neighbors],
                neighbors=neighbors,
                dE_forward=self.connectivity_table.df[
                    (self.connectivity_table.df["state"] == from_state)
                    & (self.connectivity_table.df["state_connexion"] == exit_state)
                ].iloc[0]["dE_forward"],
                k_tot=self.connectivity_table.df.loc[
                    self.connectivity_table.df["transient"] == False, "k_forward"
                ].sum(),
                t_exit=t_exit,
                exit_state=exit_state,
                from_state=from_state,
                num_reference_event=event_idx,
            )
        )

    def _initialize(self, system) -> None:
        """
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.current_state = 0
        self.states_to_explore = [0]
        self.explored_states = []
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(
            params=self.params, reference_table=self.reference_table
        )
        self.selector = FPTASelector()
        new_system = System.from_configuration(
            system.configuration.copy(), pbc=system.pbc.copy()
        )
        self._add_state(
            state_index=0, system=new_system
        )  # add current state 0 to self.states

    def construct_connexion_table(self):
        """
        explore the basin and construct the connextion table
        """
        # Loop over state to explore
        while len(self.states_to_explore) != 0:
            # next state to explore :
            to_explore = self.states_to_explore[0]
            atom_shapes = None

            if (
                to_explore not in self.states
            ):  # always true except at the start (to_explore = 0)
                # We need to create the state
                # find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = (
                    self.connectivity_table.get_transition_to_state(
                        target_state=to_explore
                    )
                )

                # Create new system by applying (reconstruction) the generic event to the from_state
                result = self.system_from_state(
                    from_state, event_idx, central_atom, sym_idx
                )
                if not result.is_ok():
                    return result
                new_system = result.ok_value()

                # Check if it is a new_system or already in states
                is_new_state = self.is_new_state(new_system)
                if is_new_state != -1:  # It already exists
                    # update table
                    self.connectivity_table.change_state_index(
                        current_index=to_explore, new_index=is_new_state
                    )
                    self.explored_states.append(to_explore)
                    self.states_to_explore.remove(to_explore)

                    # Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue  # Skip the rest

                # add state
                self._add_state(
                    state_index=to_explore, system=new_system, transient=is_transient
                )

                # ENSURE FULL STATE TO EXPLORE
                self.states[to_explore].ensure_full_state(self.params)
                # Check if unknown atomic environments
                is_unknown, atom_shapes = self.is_states_has_unknown_environments(
                    self.states[to_explore]
                )
                if is_unknown:
                    # We consider that this state is an absorbing one because we need to search new events (in main KMC loop)
                    # Need to update the connectivity table
                    self.connectivity_table.change_state_to_absorbing(to_explore)
                    self.states[to_explore].transient = False
                    is_transient = False

                if not is_transient:
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)

                    # Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()

                    continue  # We dont explore/skip the rest

                # Release heavy objet memory
                self.states[from_state].release_heavy_objects()

            # Explore state
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()

            # Ensure full state to explore
            self.states[to_explore].ensure_full_state(self.params)
            if atom_shapes is None:
                # State 0 never goes through the unknown-environments check
                # above (it's checked once, separately, before this loop
                # starts) -- resolve its atom shapes here instead.
                _, atom_shapes = self.is_states_has_unknown_environments(
                    self.states[to_explore]
                )
            self.explorer.explore(
                state=self.states[to_explore],
                state_index=self.current_state,
                start_index=last_state_connectivity,
                atom_shapes=atom_shapes,
            )

            # to_explore has been explored :
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            # Merge state connectivity table to basin connectivity table
            self.connectivity_table.merge(self.explorer.connectivity_table)
            # Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            # Clean heaby state object :
            self.states[to_explore].release_heavy_objects()

        return Ok(None)

    def select_event(self):
        """
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self):
        """
        convinient method
        """
        pass

    def get_last_state_index(self):
        if self.current_state == 0:  # connextion table is empty
            new_state_connexion = 1
        else:  # last state connexion +1
            new_state_connexion = int(
                self.connectivity_table.get_table()["state_connexion"].iloc[-1] + 1
            )
        return new_state_connexion

    def update_to_explore(self):
        # Find all state index in the connexion table :
        unique_states = set(self.connectivity_table.get_table()["state"]).union(
            set(self.connectivity_table.get_table()["state_connexion"])
        )
        self.states_to_explore = list(
            unique_states.difference(set(self.explored_states))
        )

    def system_from_state(self, from_state, event_idx, central_atom, sym_idx):
        """Reconstruct the generic event to generate new state from state"""

        ref_event = self.reference_table.table[
            self.reference_table.table["idx_ref"] == event_idx
        ]  # event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
        #        ref_event = self.reference_table.table.iloc[event_idx].copy()

        initial_configuration = ref_event["initial_configuration"]
        final_configuration = ref_event["final_configuration"]
        saddle_configuration = ref_event["saddle_configuration"]

        # Apply the generic event to the current state

        # ENSURE FULL STATE FOR FROM STATE
        self.states[from_state].ensure_full_state(self.params)

        # We start from the from_state
        new_system = System.from_configuration(
            self.states[from_state].system.configuration.copy(), pbc=True
        )
        # new_system = copy.deepcopy(self.states[from_state].system)

        # Apply PSR between event initial position and environment positions of the central_atoms
        result = PointSetRegistration(
            self.params,
            new_system,
            ref_event,
            self.states[from_state].neighbors_list,
            central_atom,
        ).match()
        if not result.is_ok():  # PSR Err
            return result
            # Check if PointSetRegistration match is valid
        result = check_match(result, self.params.psr.matching_score_thr)
        if not result.is_ok():  # PSR matching score not valid :
            return result
        else:
            psr_output = result.ok_value()  # get psr results

        # Apply PSR to generic event to move

        # Apply symmetry matrix if sym != 0
        if sym_idx != 0:
            sym_matrix = ref_event["sym_matrix"][sym_idx]
            sym_perm = ref_event["sym_perm"][sym_idx]
            # sym_matrix is a rotation about the reference initial shape's own
            # centroid (ira_mod.SOFI's convention, see unique_symmetries), not
            # the coordinate origin -- pivot on that centroid rather than the
            # raw absolute positions, or the reconstructed shape comes out
            # wrong by roughly the cluster's offset from the origin.
            pivot = initial_configuration.positions.mean(axis=0)
            initial_configuration = geometry.transform_positions(
                initial_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
            saddle_configuration = geometry.transform_positions(
                saddle_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
            final_configuration = geometry.transform_positions(
                final_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
        initial_configuration = geometry.transform_positions(
            initial_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        saddle_configuration = geometry.transform_positions(
            saddle_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        final_configuration = geometry.transform_positions(
            final_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )

        # Move system do saddle positions
        neighbors = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )

        if self.params.basin.style == "global":
            new_system.update_positions(final_configuration, atom_idx=neighbors)
            min2_configuration, _ = self.manager.global_minimize_with_results(
                self.params, configuration=new_system.configuration.copy()
            )
            new_system.update_positions(min2_configuration)

        elif self.params.basin.style == "global/reconstruction":
            new_system.update_positions(saddle_configuration, atom_idx=neighbors)

            # Reconstruct the event
            # future = self.manager.minimize_with_results(self.params, positions=new_system.positions)
            # min_pos, _ = future.result()

            local_types = np.asarray(new_system.types)[neighbors]
            supposed_initial = initial_configuration.with_types(local_types)
            supposed_final = final_configuration.with_types(local_types)
            result = Reconstruction(self.params, self.manager).reconstruct(
                supposed_initial,
                supposed_final,
                new_system.configuration,
                self.params.psr.matching_score_thr,
                neighbors,
            )
            if not result.is_ok():
                return result
            new_system.update_positions(result.ok_value().min2_configuration)

        else:
            raise ValueError(f"Unknown {self.params.basin.style} style parameter.")

        return Ok(new_system)

    def refine_absorbing(self, system):
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        # compute the energy of the state
        # for all row in connectivity table where we need to refine
        futures_context = {}  # idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.iterrows():
            if row["transient"] == False:  # need to refine
                # tmp_system = copy.deepcopy(self.states[row["state"]].system)
                tmp_system = System.from_configuration(
                    self.states[row["state"]].system.configuration.copy(), pbc=True
                )
                # get tmp_system energy
                future1 = self.manager.get_total_energy(
                    positions=tmp_system.positions.copy()
                )  # Send copy not reference
                # move to generic saddle positions
                ref_event = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == row["event_connexion"]
                ]
                if ref_event.empty:
                    raise ValueError(
                        f"idx_ref={row['event_connexion']} not found in reference table"
                    )
                ref_event = ref_event.iloc[0].copy()
                # ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
                saddle_configuration = ref_event["saddle_configuration"]
                # Apply PSR between event initial position and environment positions of the central_atoms

                # ENSURE "STATE" FULL
                self.states[row["state"]].ensure_full_state(self.params)

                result = PointSetRegistration(
                    self.params,
                    tmp_system,
                    ref_event,
                    self.states[row["state"]].neighbors_list,
                    row["central_atom"],
                ).match()
                if not result.is_ok():  # PSR Err
                    return result
                    # Check if PointSetRegistration match is valid
                matching_score_thr = self.params.psr.matching_score_thr + 0.25 * self.params.psr.matching_score_thr
                result = check_match(result, matching_score_thr)
                if not result.is_ok():  # PSR matching score not valid :
                    return result
                else:
                    psr_output = result.ok_value()  # get psr results

                # Apply symmetry matrix if sym != 0
                if row["sym"] != 0:
                    sym_matrix = ref_event["sym_matrix"][row["sym"]]
                    sym_perm = ref_event["sym_perm"][row["sym"]]
                    # sym_matrix is a rotation about the reference initial
                    # shape's own centroid (ira_mod.SOFI's convention, see
                    # unique_symmetries), not the coordinate origin or
                    # saddle_configuration's own centroid -- pivot on the
                    # initial shape's centroid rather than the raw absolute
                    # positions, or the reconstructed saddle comes out wrong
                    # by roughly the cluster's offset from the origin.
                    pivot = ref_event["initial_configuration"].positions.mean(axis=0)
                    saddle_configuration = geometry.transform_positions(
                        saddle_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
                    ) + pivot
                saddle_configuration = geometry.transform_positions(
                    saddle_configuration,
                    psr_output.rotation_matrix,
                    psr_output.translation_matrix,
                    psr_output.permutation_matrix,
                )
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors(
                    "rcut", row["central_atom"]
                )

                if self.params.control.active_volume == True:
                    # add a job to manager queue
                    future2 = self.manager.partn_refine(
                        self.params,
                        row["central_atom"],
                        configuration=tmp_system.configuration.copy(),
                        saddle_idx=neighbors.copy(),
                        saddle_positions=saddle_configuration.positions.copy(),
                    )
                # Move system do saddle positions
                else:
                    tmp_system.update_positions(saddle_configuration, atom_idx=neighbors)
                    # refine
                    future2 = self.manager.partn_refine(
                        self.params,
                        row["central_atom"],
                        configuration=tmp_system.configuration.copy(),
                        saddle_idx=neighbors.copy(),
                    )  # send copy not reference !

                # save future in context :
                futures_context[idx] = {
                    "min": future1,
                    "saddle": future2,
                    "neighbors": neighbors,
                    "state": row["state"],
                    "central_atom": row["central_atom"],
                }

                # RELEASE MEMORY :
                self.states[row["state"]].release_heavy_objects()

        # modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            E_min = ctx["min"].result()
            result_sad = ctx["saddle"].result()
            if not result_sad.is_ok():
                return result_sad
            E_sad = result_sad.ok_value().E_saddle
            if self.params.control.active_volume == True:
                dE = E_sad
            else:
                dE = E_sad - E_min
            k = compute_rate_Eyring(dE, self.params)

            # also save saddle configuration refined -- re-imaged around the
            # live central atom so every atom in the cluster is mutually
            # contiguous (a raw slice can leave an atom that drifted across
            # a periodic boundary during the saddle search off by a whole
            # cell vector relative to the rest of the cluster), matching how
            # every other local cluster in the codebase is normalized.
            idx_state = self.connectivity_table.df.loc[idx].at["state_connexion"]
            from_state_for_saddle = self.connectivity_table.df.loc[idx].at["state"]
            central_atom_position = self.states[ctx["state"]].system.positions[
                ctx["central_atom"]
            ]
            self.absorbing_saddle_configurations[(from_state_for_saddle, idx_state)] = (
                geometry.unwrap_around(
                    result_sad.ok_value().saddle[ctx["neighbors"]], central_atom_position
                )
            )
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        return Ok(None)

    def is_new_state(self, system):
        # Loop over all other system in self.states to see if system is already known

        for state_index, state_data in self.states.items():
            are_equivalent = self.are_structures_equivalent(
                system.configuration, state_data.system.configuration
            )
            if are_equivalent:
                return state_index
        return -1

    def are_structures_equivalent(
        self, configuration1: Configuration, configuration2: Configuration, tol=0.3
    ):
        pos1, pos2 = configuration1.positions, configuration2.positions
        if len(pos1) != len(pos2):
            return False

        # In full coloring mode, two states with the same geometry but a different
        # species arrangement (e.g. an Fe/Ni swap) are distinct; in grey mode they merge.
        if self.params.atomicenvironment.coloring_mode == "full" and not np.array_equal(
            configuration1.types, configuration2.types
        ):
            return False

        box = np.diag(configuration1.cell).tolist()
        tree2 = cKDTree(pos2, boxsize=box)
        distances, _ = tree2.query(pos1, k=1)

        return np.max(distances) < tol

    def is_states_has_unknown_environments(
        self, state: StateData
    ) -> tuple[bool, dict[int, ShapeID]]:
        """Check whether any atom in `state` has a shape not already catalogued.

        An atom classified `"crystal"` needs no event and is treated as
        known without classification: under the `cna/graph`-family styles,
        a bulk/perfectly-coordinated atom never gets a real topology hash
        computed for it in the first place (see
        `AtomicEnvironment.compute_cnagraph`), so there is no shape for it
        to classify into.

        Every other atom must resolve to a catalogued `ShapeID`: a coarse
        `id_initial` collision can leave an atom's real shape uncatalogued
        even though its `id_initial` has other, unrelated rows in
        `reference_table` -- so this classifies each remaining atom's live
        shape directly (`ShapeTable.classify_live_sid`)
        rather than checking `id_initial` presence alone. Basin exploration
        can only build connectivity from already-catalogued events (no live
        ARTn search happens here), so any atom that fails to classify means
        this state must be treated as absorbing and handed back to the main
        KMC loop, which can actually search it.

        Parameters
        ----------
        state : StateData
            The state to check.

        Returns
        -------
        bool
            True if at least one non-`"crystal"` atom fails to classify into
            a catalogued shape under its `id_initial` (including an
            `id_initial` with no rows at all).
        dict[int, ShapeID]
            Every atom classified before the answer was decided. Complete
            (covers every non-crystal atom) when the bool is `False`, so the
            caller can pass it straight to `BasinGenericEventExplorer.explore()`
            instead of reclassifying; partial (stops at the first miss) and
            unused by the caller when the bool is `True`.

        """
        atom_shapes: dict[int, ShapeID] = {}
        for atom_idx, id_initial in enumerate(
            state.environment.atomic_environment_list
        ):
            if id_initial == "crystal":
                continue
            shape = self.reference_table.shapes.resolve_live_shape(
                state.system, state.neighbors_list, state.environment, atom_idx, mint=False
            )
            if shape is None:
                return True, atom_shapes
            atom_shapes[atom_idx] = shape
        return False, atom_shapes

    def _add_state(
        self,
        state_index,
        system=None,
        transient=True,
        applicable_events=None,
        visited=False,
        full=False,
    ):
        """Add a new state in the `self.states` dictionnary."""
        # to fit typing
        neighbors_list = []
        atomic_environment = []

        if full == True:
            neighbors_list = NeighborsList(
                system,
                self.params.atomicenvironment.rnei,
                self.params.atomicenvironment.rcut,
                self.params.atomicenvironment.rnei_pairs,
            )
            atomic_environment = AtomicEnvironment(
                self.params.atomicenvironment.style,
                neighbors_list.neighbors_list["rnei"],
                neighbors_list.neighbors_list["rcut"],
                self.params.atomicenvironment.neighbors_add,
                coordination_threshold=self.params.atomicenvironment.coordination_threshold,
                types=system.types,
                coloring_mode=self.params.atomicenvironment.coloring_mode,
            )
        else:
            neighbors_list = None
            atomic_environment = None
        new_state = StateData(
            system=system,
            environment=atomic_environment,
            neighbors_list=neighbors_list,
            transient=transient,
            visited=visited,
        )

        self.states[state_index] = new_state
