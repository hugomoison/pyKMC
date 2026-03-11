from .exploration import BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass
from pykmc import System, Config, NeighborsList, AtomicEnvironment, PointSetRegistration, check_match
from typing import Optional
from ..utils import geometry
from ..rate_constant import compute_rate_Eyring
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, Err, BasinOutput, ErrorInfo, ErrorType
import logging
import time

logger = logging.getLogger("log")

#TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
#TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
#TODO: Think about parallized exploration 
#TODO: Could think of refining transient -> absorbing event when exploring
#TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
#TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.

@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList] 
    transient: bool = False
    visited: bool = False

    def release_heavy_objects(self) -> None : 
        """Release heavy objects"""
        self.neighbors_list = None 
        self.environment = None
    
    def ensure_full_state(self, config: Config) -> None : 
        if self.system is not None : 
            if self.neighbors_list is None : 
                self.neighbors_list = NeighborsList(self.system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)  
            if self.environment is None :
                types = self.system.types if config.atomicenvironment.atom_coloring_mode == "full" else None
                self.environment = AtomicEnvironment(config.atomicenvironment.style, self.neighbors_list.neighbors_list["rnei"], self.neighbors_list.neighbors_list["rcut"], config.atomicenvironment.neighbors_add, types=types, coordination_threshold=config.atomicenvironment.coordination_threshold)


class BasinsGenericEvents() : 

    def __init__(self, config: Config, reference_table,known_environments, manager ) -> None :  
        self.config = config #Config object with basins parameters
        self.explorer = None #object to explore a state in the basin 
        self.reference_table = reference_table #Object with reference generic events
        self.manager = manager #object to do external task (minimize, refine)

        self.connectivity_table = None #Dataframe of basin connexion state
        self.selected_event = None #The selected event after basin exploration
        self.current_state = None #Current state where we're at
        self.states_to_explore = None #List of state to explore
        self.explored_states = None #List of state that we already explored
        self.states: dict[int, StateData] = {}  #Dictionnary of StateDate
        self._state_fingerprints: dict[int, np.ndarray] = {}  # Fast dedup rejection cache
        self.known_environments = known_environments
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}
        self._next_state_index = 1  # Monotonic counter for state indices (0 is the initial state)
        self._use_session_pool = False  # Set True only for parallel strategies that call use_local()
        self._was_capped = False

    def detection(self, params) -> bool : 
        """Utility method."""
        return self.detector.detection(**params) 
    
    def execute(self, system) : 
        """ 
        Run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        self._initialize(system)
        result = self._construct_connectivity_table()
        if not result.is_ok() :
            return result

        self._finalize_connectivity_table()
        result = self._refine_absorbing_states(system)
        if not result.is_ok() :
            return result

        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok() :
            return result
        return self._build_basin_output(result.ok_value())

    def _construct_connectivity_table(self):
        strategy = self.config.basin.strategy
        if strategy == "serial":
            return self.construct_connexion_table()
        return self.construct_connexion_table_parallel()

    def _connectivity_state_counts(self) -> tuple[int, int]:
        if self.connectivity_table.df.empty:
            return 0, 0
        transient_states = set(self.connectivity_table.df["state"])
        all_states = transient_states | set(self.connectivity_table.df["state_connexion"])
        return len(transient_states), len(all_states) - len(transient_states)

    def _finalize_connectivity_table(self) -> None:
        table_states = set(self.connectivity_table.df["state"]) | set(self.connectivity_table.df["state_connexion"])
        missing_from_states = table_states - set(self.states.keys())
        if missing_from_states:
            raise RuntimeError(
                f"[Basin] BUG: {len(missing_from_states)} states in connectivity table but not in self.states: "
                f"{sorted(missing_from_states)[:10]}..."
            )

        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}

        n_transient, n_absorbing_states = self._connectivity_state_counts()
        self.connectivity_table.df["transient"] = self.connectivity_table.df["state_connexion"].apply(lambda x: x < n_transient)
        n_absorbing_rows = len(self.connectivity_table.df[self.connectivity_table.df["transient"] == False])
        logger.info(
            "[Basin] Reordered: %d transient + %d absorbing states | %d absorbing rows to refine",
            n_transient,
            n_absorbing_states,
            n_absorbing_rows,
        )

    def _refine_absorbing_states(self, system):
        self.manager.use_local()
        try:
            result = self.refine_absorbing(system)
        finally:
            self.manager.use_global()
        if result.is_ok():
            logger.info("[Basin] Refined %d absorbing states", len(self.absorbing_saddle_positions))
        return result

    def _build_basin_output(self, selection_output):
        t_exit = selection_output.t_exit
        exit_state = selection_output.exit_state
        logger.info("[Basin] FPTA selected: exit_state=%d, t_exit=%.6e", exit_state, t_exit)

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=exit_state)
        self.states[from_state].ensure_full_state(self.config)

        neighbors = self.states[from_state].neighbors_list.get_neighbors("rcut", central_atom)
        return Ok(BasinOutput(initial_system_positions=self.states[from_state].system.positions, 
                              central_atom=central_atom, 
                              saddle_positions=self.absorbing_saddle_positions[exit_state], 
                              final_positions=self.states[exit_state].system.positions[neighbors], 
                              neighbors=neighbors,
                              energy_barrier= self.connectivity_table.df[(self.connectivity_table.df["state"] == from_state) & (self.connectivity_table.df["state_connexion"] == exit_state)].iloc[0]["dE_forward"], 
                              k_tot = self.connectivity_table.df.loc[self.connectivity_table.df["transient"] == False, "k_forward"].sum(),
                              t_exit = t_exit,
                              exit_state = exit_state, 
                              from_state = from_state,
                              num_reference_event= event_idx))
        

    def _initialize(self, system) -> None: 
        """ 
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.current_state = 0
        self.selected_event = None
        self.states_to_explore = [0]
        self.explored_states = []
        self.states = {}
        self._state_fingerprints = {}
        self.absorbing_saddle_positions = {}
        self._next_state_index = 1  # State 0 is already assigned
        self._was_capped = False
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = FPTASelector()
        new_system = System(positions=system.positions.copy(), types=system.types.copy(), cell=system.cell.copy(), pbc=system.pbc.copy(), index=np.arange(len(system.types)))
        self._add_state(state_index=0, system=new_system)  #add current state 0 to self.states

    def _finalize_exploration_run(self, t_start, prof, n_processed, n_duplicates, strategy_label: Optional[str] = None):
        elapsed = time.perf_counter() - t_start
        n_transient, n_absorbing_final = self._connectivity_state_counts()
        label = f" ({strategy_label})" if strategy_label else ""
        logger.info(
            "[Basin] COMPLETE%s: %d transient + %d absorbing states | %d connectivity rows | processed=%d | duplicates=%d | %.1fs",
            label,
            n_transient,
            n_absorbing_final,
            len(self.connectivity_table.df),
            n_processed,
            n_duplicates,
            elapsed,
        )

        top_level = {k: v for k, v in prof.items() if k not in ("other", "psr", "minimize")}
        prof["other"] = elapsed - sum(top_level.values())
        logger.info(
            "[Basin] PROFILING: reconstruct=%.2fs (psr=%.2fs + min=%.2fs) | dedup=%.2fs | explore=%.2fs | ensure_state=%.2fs | merge=%.2fs | other=%.2fs | total=%.2fs",
            prof["reconstruct"],
            prof["psr"],
            prof["minimize"],
            prof["dedup"],
            prof["explore"],
            prof["ensure_state"],
            prof["merge"],
            prof["other"],
            elapsed,
        )
        for phase, t in sorted(top_level.items(), key=lambda x: -x[1]):
            pct = 100.0 * t / elapsed if elapsed > 0 else 0
            logger.info("[Basin] PROFILING:   %-15s %8.2fs  %5.1f%%", phase, t, pct)

        self._write_timing_checkpoint(prof, elapsed, n_transient, n_absorbing_final, n_duplicates, n_processed)
        return Ok(None)

    def construct_connexion_table(self) :
        """Explore the basin and construct the connextion table
        """
        import time
        t_start = time.perf_counter()
        n_explored = 0
        n_duplicates = 0
        n_absorbing = 0
        n_processed = 0

        # Profiling accumulators (per-phase wall time in seconds)
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        # Switch to session pool for basin reconstruction (parallel minimization)
        self.manager.use_local()

        max_states: int | None = self.config.basin.max_states
        t_last_log = t_start

        try:
            #Loop over state to explore
            while len(self.states_to_explore) != 0 :
                # Check max_states cap
                if max_states is not None and n_explored >= max_states:
                    logger.warning("[Basin] max_states=%d reached. Capping.", max_states)
                    result = self._cap_remaining_as_absorbing()
                    if not result.is_ok():
                        return result
                    break

                #next state to explore :
                to_explore = self.states_to_explore[0]

                if to_explore not in self.states : #always true except at the start (to_explore = 0)
                    n_processed += 1
                    #We need to create the state
                        #find a state and an event from which we go to the state that we want to create
                    from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)

                        #Create new system by applying (reconstruction) the generic event to the from_state
                    t0 = time.perf_counter()
                    result = self.system_from_state(from_state, event_idx, central_atom, sym_idx)
                    prof["reconstruct"] += time.perf_counter() - t0
                    if not result.is_ok() :
                        return result
                    new_system = result.ok_value()

                        #Check if it is a new_system or already in states
                    t0 = time.perf_counter()
                    is_new_state = self.is_new_state(new_system)
                    prof["dedup"] += time.perf_counter() - t0
                    if is_new_state != -1 : #It already exists
                        #update table
                        self.connectivity_table.change_state_index(current_index=to_explore, new_index=is_new_state)
                        self.explored_states.append(to_explore)
                        self.states_to_explore.remove(to_explore)
                        n_duplicates += 1

                        if n_duplicates % 20 == 0:
                            elapsed = time.perf_counter() - t_start
                            logger.debug("[Basin] processed=%d | duplicates=%d | absorbing=%d | explored=%d | to_explore=%d | %.1fs",
                                         n_processed, n_duplicates, n_absorbing, n_explored, len(self.states_to_explore), elapsed)

                        #Cleaning
                        self.states[from_state].release_heavy_objects()
                        continue #Skip the rest

                    #add state
                    self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                    #Ensure full state to explore
                    t0 = time.perf_counter()
                    self.states[to_explore].ensure_full_state(self.config)
                    prof["ensure_state"] += time.perf_counter() - t0
                    #Check if unknown atomic environments
                    if self.is_states_has_unknown_environments(self.states[to_explore]) :
                        #We consider that this state is an absorbing one because we need to search new events (in main KMC loop)
                        #Need to update the connectivity table
                        self.connectivity_table.change_state_to_absorbing(to_explore)
                        self.states[to_explore].transient = False
                        is_transient = False

                    if not is_transient :
                        self.states_to_explore.remove(to_explore)
                        self.explored_states.append(to_explore)
                        n_absorbing += 1

                        if n_absorbing % 20 == 0:
                            elapsed = time.perf_counter() - t_start
                            logger.debug("[Basin] processed=%d | absorbing=%d | duplicates=%d | explored=%d | to_explore=%d | %.1fs",
                                         n_processed, n_absorbing, n_duplicates, n_explored, len(self.states_to_explore), elapsed)

                        #Cleaning
                        self.states[from_state].release_heavy_objects()
                        self.states[to_explore].release_heavy_objects()

                        continue #We dont explore/skip the rest

                    #Release heavy objet memory
                    self.states[from_state].release_heavy_objects()


                #Explore state via MPI engine
                self.current_state = to_explore
                last_state_connectivity = self.get_last_state_index()

                t0 = time.perf_counter()
                self._explore_states_parallel([to_explore], n_workers=1)
                prof["explore"] += time.perf_counter() - t0

                #to_explore has been explored :
                self.states_to_explore.remove(to_explore)
                self.explored_states.append(to_explore)

                t0 = time.perf_counter()
                self.update_to_explore()
                prof["merge"] += time.perf_counter() - t0
                #Clean heavy state object :
                self.states[to_explore].release_heavy_objects()

                # Progress tracking
                n_explored += 1
                elapsed = time.perf_counter() - t_start
                logger.debug("[Basin] explored=%d | to_explore=%d | unique_states=%d | duplicates=%d | absorbing=%d | conn_rows=%d | %.1fs",
                             n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, len(self.connectivity_table.df), elapsed)

                # Periodic progress log (every 30s)
                now = time.perf_counter()
                if now - t_last_log >= 30.0:
                    logger.info(
                        "[Basin] PROGRESS: explored=%d | to_explore=%d | states=%d | duplicates=%d | absorbing=%d | %.0fs elapsed",
                        n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, now - t_start,
                    )
                    t_last_log = now
        finally:
            self.manager.use_global()
        return self._finalize_exploration_run(t_start, prof, n_processed, n_duplicates)

    def select_event(self) :
        """ 
        Select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        Convinient method
        """
        pass

    def get_last_state_index(self) :
        """Return the next available state index (monotonically increasing).

        Using a monotonic counter prevents index reuse when change_state_index()
        remaps high-valued indices to lower ones, which would cause the table max
        to drop and subsequent explorations to reuse indices already in explored_states.
        """
        return self._next_state_index
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()["state"]).union(set(self.connectivity_table.get_table()["state_connexion"]))
        self.states_to_explore =  list(unique_states.difference(set(self.explored_states)))


    def _prepare_reconstruct_kwargs(self, from_state, event_idx, central_atom, sym_idx):
        """Prepare keyword arguments for manager.basin_reconstruct().

        Gathers all data needed by the engine to perform PSR + minimize.
        """
        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx]
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()

        self.states[from_state].ensure_full_state(self.config)
        neighbor_indices = self.states[from_state].neighbors_list.get_neighbors("rcut", central_atom)

        return {
            "config": self.config,
            "from_positions": self.states[from_state].system.positions.copy(),
            "from_types": list(self.states[from_state].system.types),
            "cell": self.states[from_state].system.cell.copy(),
            "pbc": self.states[from_state].system.pbc,
            "ref_initial_positions": np.array(ref_event["initial_positions"], copy=True),
            "ref_saddle_positions": np.array(ref_event["saddle_positions"], copy=True),
            "ref_final_positions": np.array(ref_event["final_positions"], copy=True),
            "ref_initial_types": ref_event.get("initial_types"),
            "sym_matrices": ref_event["sym_matrix"],
            "sym_perms": ref_event["sym_perm"],
            "central_atom": central_atom,
            "sym_idx": sym_idx,
            "neighbor_indices": neighbor_indices,
            "matching_score_thr": self.config.psr.matching_score_thr,
            "kmax_factor": self.config.ira.kmax_factor,
            "atom_coloring_mode": self.config.atomicenvironment.atom_coloring_mode,
        }

    def _result_from_mpi(self, mpi_result, from_state):
        """Convert MPI basin_reconstruct result dict to Ok(System) or Err(ErrorInfo)."""
        if mpi_result is None or not mpi_result.get("ok"):
            error_type_str = mpi_result.get("error_type", "UNKNOWN") if mpi_result else "UNKNOWN"
            message = mpi_result.get("message", "Unknown error") if mpi_result else "No result from engine"
            error_type = getattr(ErrorType, error_type_str, ErrorType.RECONSTRUCTION_INVALID_MIN2)
            return Err(ErrorInfo(type=error_type, message=message))

        import ase.geometry
        cell = self.states[from_state].system.cell
        pbc = self.states[from_state].system.pbc
        positions = ase.geometry.wrap_positions(
            positions=mpi_result["min2_positions"], cell=cell, pbc=pbc)
        new_system = System(
            positions=positions,
            types=self.states[from_state].system.types,
            cell=cell,
            pbc=pbc,
            index=np.arange(len(self.states[from_state].system.types)))
        return Ok(new_system)

    def _transport_error(self, operation: str, exc: Exception):
        return Err(
            ErrorInfo(
                type=ErrorType.MPI_REMOTE_ERROR,
                message=f"{operation} failed: {exc}",
            )
        )

    def system_from_state(self, from_state, event_idx, central_atom, sym_idx):
        """Reconstruct a new state via MPI engine (PSR + minimize).

        Submits the reconstruction task to an engine rank and blocks until complete.
        """
        kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
        future = self.manager.basin_reconstruct(**kwargs)
        try:
            mpi_result = future.result()
        except Exception as exc:
            return self._transport_error("basin_reconstruct", exc)
        return self._result_from_mpi(mpi_result, from_state)

    def _materialize_frontier_state(self, state_idx: int):
        """Reconstruct a frontier state so it can be treated as an absorbing exit."""
        if state_idx in self.states:
            return Ok(state_idx)

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=state_idx)
        result = self.system_from_state(from_state, event_idx, central_atom, sym_idx)
        if not result.is_ok():
            return result

        new_system = result.ok_value()
        existing_state = self.is_new_state(new_system)
        if existing_state != -1:
            self.connectivity_table.change_state_index(current_index=state_idx, new_index=existing_state)
            return Ok(existing_state)

        self._add_state(state_index=state_idx, system=new_system, transient=is_transient)
        self.states[from_state].release_heavy_objects()
        return Ok(state_idx)

    def refine_absorbing(self, system) :
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        #compute the energy of the state 
        #for all row in connectivity table where we need to refine
        futures_context = {} #idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.iterrows() : 
            if row["transient"]  == False : #need to refine
                #tmp_system = copy.deepcopy(self.states[row["state"]].system)
                tmp_system = System(positions=self.states[row["state"]].system.positions.copy(), types=self.states[row["state"]].system.types, cell=self.states[row["state"]].system.cell, pbc=self.states[row["state"]].system.pbc, index=np.arange(len(self.states[row["state"]].system.types)))
                #get tmp_system energy 
                future1 = self.manager.get_total_energy(positions=tmp_system.positions.copy()) #Send copy not reference
                #move to generic saddle positions 
                ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == row["event_connexion"]] 
                if ref_event.empty:
                    raise ValueError(f"idx_ref={row['event_connexion']} not found in reference table")
                ref_event = ref_event.iloc[0].copy()
                #ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
                saddle_positions = ref_event["saddle_positions"].copy()
                #Apply PSR between event initial position and environment positions of the central_atoms


                #ENSURE "STATE" FULL 
                self.states[row["state"]].ensure_full_state(self.config)

                result = PointSetRegistration(self.config, tmp_system, ref_event , self.states[row["state"]].neighbors_list, row["central_atom"]).match()
                if not result.is_ok(): #PSR Err
                    return result
                    # Check if PointSetRegistration match is valid 
                result = check_match(result, self.config.psr.matching_score_thr)
                if not result.is_ok() : #PSR matching score not valid : 
                    return result
                else : 
                    psr_output = result.ok_value() #get psr results

                # Apply symmetry matrix if sym != 0
                if row["sym"] != 0 :
                    sym_matrices = ref_event["sym_matrix"]
                    sym_matrix = sym_matrices[row["sym"]]
                    saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][row["sym"]])
                saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors("rcut", row["central_atom"])

                if self.config.control.active_volume==True:
                    # add a job to manager queue
                    future2 = self.manager.partn_refine(self.config, row["central_atom"],
                                                  tmp_system.positions.copy(),
                                                  tmp_system.cell,
                                                  tmp_system.types.copy(),
                                                  neighbors.copy(),
                                                  saddle_positions.copy())
                # Move system do saddle positions
                else:
                    tmp_system.update_positions(saddle_positions, atom_idx = neighbors)
                    #refine
                    future2 = self.manager.partn_refine(self.config, row["central_atom"], tmp_system.positions.copy()) #send copy not reference !
                
                #save future in context : 
                futures_context[idx] = {
            "min": future1,
            "saddle": future2, 
            "neighbors": neighbors}
                
                #RELEASE MEMORY : 
                self.states[row["state"]].release_heavy_objects()

        #modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            E_min    = ctx["min"].result()
            result_sad = ctx["saddle"].result()
            if not result_sad.is_ok() : 
                return result_sad
            E_sad = result_sad.ok_value().E_saddle
            if self.config.control.active_volume==True:
                dE = E_sad
            else:
                dE = E_sad - E_min
            k = compute_rate_Eyring(dE, self.config)

            #also save saddle positions refined 
            idx_state = self.connectivity_table.df.loc[idx].at["state_connexion"]
            central_atom = self.connectivity_table.df.loc[idx].at["central_atom"]
            #self.absorbing_saddle_positions[idx_state] = result.ok_value().saddle_positions[self.states[idx_state].neighbors_list.get_neighbors("rcut", central_atom)]
            self.absorbing_saddle_positions[idx_state] = result_sad.ok_value().saddle_positions[ctx["neighbors"]]
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        return Ok(None)


    def _fingerprint_tolerance(self) -> float:
        """Return the max element-wise diff for fingerprint pre-filtering."""
        if (self.config.basin is not None
                and self.config.basin.fingerprint_tolerance is not None):
            return self.config.basin.fingerprint_tolerance
        # COM-distance tolerance (used by full COM and coordination-COM hybrid)
        return 0.5

    def is_new_state(self, system) :
        #Loop over all other system in self.states to see if system is already known
        fp_new = self._compute_fingerprint(system.positions, system.cell, system.pbc)
        fp_tol = self._fingerprint_tolerance()

        # Vectorized fingerprint rejection: compare against all states at once
        fp_items = [
            (si, fp)
            for si, fp in self._state_fingerprints.items()
            if len(fp) == len(fp_new)
        ]
        if fp_items:
            indices, fps = zip(*fp_items)
            fp_matrix = np.vstack(fps)  # (N_states, N_atoms)
            max_diffs = np.max(np.abs(fp_matrix - fp_new[np.newaxis, :]), axis=1)
            candidates = [indices[i] for i in np.where(max_diffs <= fp_tol)[0]]
        else:
            candidates = list(self.states.keys())

        for state_index in candidates:
            state_data = self.states[state_index]
            if state_data.system is None:
                continue
            are_equivalent = self.are_structures_equivalent(system.positions, state_data.system.positions, cell = system.cell, pbc=system.pbc)
            if are_equivalent :
                return state_index
        return -1


    @staticmethod
    def _wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Wrap positions into [0, box) for cKDTree periodic queries."""
        box = np.diag(cell)
        wrapped = np.mod(positions, box)
        return wrapped

    def are_structures_equivalent(self, pos1, pos2, cell, pbc=None, tol=0.3):

        if len(pos1) != len(pos2):
            return False

        if pbc is None or np.all(pbc):
            # Fully periodic: use boxsize (existing fast path)
            box = np.diag(cell).tolist()
            wrapped1 = self._wrap_positions(pos1, cell)
            wrapped2 = self._wrap_positions(pos2, cell)
            tree2 = cKDTree(wrapped2, boxsize=box)
            distances, _ = tree2.query(wrapped1, k=1)
        else:
            # Mixed PBC: manual minimum-image distance
            box = np.diag(cell)
            distances = np.zeros(len(pos1))
            for i, p in enumerate(pos1):
                diffs = pos2 - p
                for dim in range(3):
                    if pbc[dim]:
                        diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
                distances[i] = np.min(np.linalg.norm(diffs, axis=1))

        return np.max(distances) < tol

    def is_states_has_unknown_environments(self, state: StateData) : 
        if set(state.environment.atomic_environment_list).difference(self.known_environments) != set() :
            return True 
        else : 
            return False

    @staticmethod
    def _com_fingerprint(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """Sorted per-atom distances from center of mass (legacy fallback)."""
        box = np.diag(cell).astype(np.float64)
        pbc_array = np.asarray(pbc, dtype=bool) if pbc is not None else np.array([True, True, True])
        pos = np.array(positions, dtype=np.float64, copy=True)
        for dim in range(3):
            if pbc_array[dim] and box[dim] > 0:
                pos[:, dim] = np.mod(pos[:, dim], box[dim])
        com = pos.mean(axis=0)
        diffs = pos - com
        for dim in range(3):
            if pbc_array[dim] and box[dim] > 0:
                diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
        return np.sort(np.linalg.norm(diffs, axis=1))

    @staticmethod
    def _atoms_of_interest_fingerprint(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray,
                                        rnei: float, coord_thr: int) -> np.ndarray:
        """Atoms of interest fingerprint: sorted COM distances for undercoordinated atoms.

        Identifies 'atoms of interest' — those with fewer than coord_thr neighbors
        within rnei (near defects, surfaces, vacancies). Computes their sorted
        distances from the system center of mass, producing a short, continuous-valued
        fingerprint vector with high discriminating power.
        """
        pbc_array = np.asarray(pbc, dtype=bool) if pbc is not None else np.array([True, True, True])
        cell_diag = np.diag(cell).astype(np.float64)
        pos = np.array(positions, dtype=np.float64, copy=True)

        # Wrap positions for PBC
        for dim in range(3):
            if pbc_array[dim] and cell_diag[dim] > 0:
                pos[:, dim] = np.mod(pos[:, dim], cell_diag[dim])

        # Build tree and count neighbors
        if np.all(pbc_array):
            tree = cKDTree(pos, boxsize=cell_diag.tolist())
        else:
            tree = cKDTree(pos)

        neighbor_lists = tree.query_ball_point(pos, rnei)
        counts = np.array([len(n) - 1 for n in neighbor_lists], dtype=np.int32)

        # Find interesting atom indices (undercoordinated)
        interesting_mask = counts < coord_thr
        if not np.any(interesting_mask):
            return np.array([], dtype=np.float64)

        # COM of full system, distances for interesting atoms only
        com = pos.mean(axis=0)
        diffs = pos[interesting_mask] - com
        for dim in range(3):
            if pbc_array[dim] and cell_diag[dim] > 0:
                diffs[:, dim] -= np.round(diffs[:, dim] / cell_diag[dim]) * cell_diag[dim]

        return np.sort(np.linalg.norm(diffs, axis=1))

    def _compute_fingerprint(self, positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
        """Compute a structural fingerprint for fast inequality rejection.

        Dispatches to:
        1. Atoms of interest fingerprint if fingerprint_coordination_thr is set (explicit override)
        2. Atoms of interest fingerprint if AtomicEnvironment uses coordination style
        3. Full COM-distance fingerprint otherwise
        """
        # Explicit override from [BASIN] config
        if (self.config.basin is not None
                and self.config.basin.fingerprint_coordination_thr is not None):
            return self._atoms_of_interest_fingerprint(
                positions, cell, pbc,
                rnei=self.config.atomicenvironment.rnei,
                coord_thr=self.config.basin.fingerprint_coordination_thr,
            )
        # Auto-detect from AtomicEnvironment style
        if (self.config.atomicenvironment.style in ("coordination", "coordination/graph")
                and self.config.atomicenvironment.coordination_threshold is not None):
            return self._atoms_of_interest_fingerprint(
                positions, cell, pbc,
                rnei=self.config.atomicenvironment.rnei,
                coord_thr=self.config.atomicenvironment.coordination_threshold + 1,
            )
        return self._com_fingerprint(positions, cell, pbc)

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False, full=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing
        neighbors_list  = []
        atomic_environment = []

        if full == True :
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)
            types = system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list["rnei"], neighbors_list.neighbors_list["rcut"], self.config.atomicenvironment.neighbors_add, types=types, coordination_threshold=self.config.atomicenvironment.coordination_threshold)
        else :
            neighbors_list = None
            atomic_environment = None
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state
        if system is not None:
            self._state_fingerprints[state_index] = self._compute_fingerprint(
                system.positions, system.cell, system.pbc
            )

    def _write_timing_checkpoint(self, prof, elapsed, n_transient, n_absorbing, n_duplicates, n_processed):
        """Write a timing summary file for compare_scaling.py."""
        strategy = getattr(self.config.basin, "strategy", "serial")
        n_workers = getattr(self.config.basin, "n_workers", 1)
        n_conn = len(self.connectivity_table.df) if not self.connectivity_table.df.empty else 0

        # Write as level_complete checkpoint (L0 = single-level basin)
        ckpt_path = f"basin_timing_{strategy}.txt"
        with open(ckpt_path, "w") as f:
            f.write("# Basin timing checkpoint\n")
            f.write(f"strategy = {strategy}\n")
            f.write(f"n_workers = {n_workers}\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_transient = {n_transient}\n")
            f.write(f"states_absorbing = {n_absorbing}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")
            f.write(f"n_duplicates = {n_duplicates}\n")
            f.write(f"n_processed = {n_processed}\n")
            for phase, t in sorted(prof.items(), key=lambda x: -x[1]):
                pct = 100.0 * t / elapsed if elapsed > 0 else 0
                f.write(f"prof_{phase} = {t:.3f}\n")
                f.write(f"pct_{phase} = {pct:.1f}\n")
        logger.info("[Basin] Timing checkpoint written to %s", ckpt_path)

        # Also write as level_complete format for compare_scaling.py compatibility
        level_path = "basin_connectivity_0_L0_level_complete.txt"
        with open(level_path, "w") as f:
            f.write("# Basin level complete checkpoint\n")
            f.write("level = 0\n")
            f.write(f"wall_time_s = {elapsed:.3f}\n")
            f.write(f"level_wall_time_s = {elapsed:.3f}\n")
            f.write(f"states_total = {n_transient + n_absorbing}\n")
            f.write(f"connectivity_rows = {n_conn}\n")

    def _cap_remaining_as_absorbing(self):
        """Convert all remaining frontier states to absorbing and clear the queue."""
        self._was_capped = True
        capped = list(self.states_to_explore)
        for state_idx in capped:
            result = self._materialize_frontier_state(state_idx)
            if not result.is_ok():
                return result

            materialized_idx = result.ok_value()
            if materialized_idx != state_idx:
                if state_idx not in self.explored_states:
                    self.explored_states.append(state_idx)
                continue

            self.connectivity_table.change_state_to_absorbing(state_idx)
            if state_idx in self.states:
                self.states[state_idx].transient = False
                self.states[state_idx].release_heavy_objects()
            if state_idx not in self.explored_states:
                self.explored_states.append(state_idx)
        self.states_to_explore.clear()
        logger.warning(
            "[Basin] Capped %d remaining frontier states as absorbing.",
            len(capped),
        )
        return Ok(None)

    # ──────────────────────────────────────────────────────────────────
    # Parallel basin exploration strategies
    # ──────────────────────────────────────────────────────────────────

    def _prepare_explore_kwargs(self, state_idx, start_index):
        """Prepare keyword arguments for manager.basin_explore()."""
        import pickle

        self.states[state_idx].ensure_full_state(self.config)
        state = self.states[state_idx]

        config_dict = {
            "rnei": self.config.atomicenvironment.rnei,
            "rcut": self.config.atomicenvironment.rcut,
            "neighbors_add": self.config.atomicenvironment.neighbors_add,
            "ae_style": self.config.atomicenvironment.style,
            "atom_coloring_mode": self.config.atomicenvironment.atom_coloring_mode,
            "coordination_threshold": self.config.atomicenvironment.coordination_threshold,
            "energy_thr": self.config.basin.energy_thr,
        }

        return {
            "config_dict": config_dict,
            "reference_table_data": pickle.dumps(self.reference_table.table),
            "state_positions": state.system.positions.copy(),
            "state_types": list(state.system.types),
            "state_cell": state.system.cell.copy(),
            "state_pbc": state.system.pbc,
            "state_index": state_idx,
            "start_index": start_index,
        }

    def _reconstruct_wavefront_batch(self, to_reconstruct):
        reconstructed = {}
        transition_info = {}
        if not to_reconstruct:
            return Ok((reconstructed, transition_info, 0))

        for state_idx in to_reconstruct:
            transition_info[state_idx] = self.connectivity_table.get_transition_to_state(target_state=state_idx)

        futures = {}
        for state_idx in to_reconstruct:
            from_state, event_idx, central_atom, sym_idx, is_transient = transition_info[state_idx]
            kwargs = self._prepare_reconstruct_kwargs(from_state, event_idx, central_atom, sym_idx)
            futures[state_idx] = (from_state, self.manager.basin_reconstruct(**kwargs))

        reconstruction_error = None
        for state_idx, (from_state, future) in futures.items():
            try:
                mpi_result = future.result()
            except Exception as exc:
                result = self._transport_error("basin_reconstruct", exc)
                logger.warning("[Basin] Reconstruction transport failed for state %d: %s", state_idx, result.err_value())
                if reconstruction_error is None:
                    reconstruction_error = result
                continue
            result = self._result_from_mpi(mpi_result, from_state)
            if result.is_ok():
                reconstructed[state_idx] = result.ok_value()
                continue

            logger.warning("[Basin] Reconstruction failed for state %d: %s", state_idx, result.err_value())
            if reconstruction_error is None:
                reconstruction_error = result

        if reconstruction_error is not None:
            return reconstruction_error
        return Ok((reconstructed, transition_info, len(to_reconstruct)))

    def _register_wavefront_states(self, to_reconstruct, reconstructed, transition_info, prof):
        if len(reconstructed) > 1:
            dedup_results = self.is_new_state_batch(reconstructed)
        elif len(reconstructed) == 1:
            dedup_results = {}
            for state_idx, system in reconstructed.items():
                dedup_results[state_idx] = self.is_new_state(system)
        else:
            dedup_results = {}

        new_transient = []
        n_duplicates = 0
        n_absorbing = 0
        for state_idx in to_reconstruct:
            existing = dedup_results.get(state_idx, -1)
            if existing != -1:
                self.connectivity_table.change_state_index(current_index=state_idx, new_index=existing)
                self.explored_states.append(state_idx)
                self.states_to_explore.remove(state_idx)
                n_duplicates += 1
                continue

            is_transient = transition_info[state_idx][4]
            self._add_state(state_index=state_idx, system=reconstructed[state_idx], transient=is_transient)

            t0 = time.perf_counter()
            self.states[state_idx].ensure_full_state(self.config)
            prof["ensure_state"] += time.perf_counter() - t0

            if self.is_states_has_unknown_environments(self.states[state_idx]):
                self.connectivity_table.change_state_to_absorbing(state_idx)
                self.states[state_idx].transient = False
                is_transient = False

            if not is_transient:
                self.states_to_explore.remove(state_idx)
                self.explored_states.append(state_idx)
                self.states[state_idx].release_heavy_objects()
                n_absorbing += 1
            else:
                new_transient.append(state_idx)

        return new_transient, n_duplicates, n_absorbing

    def _explore_states_parallel(self, states_batch, n_workers=4):
        """Explore multiple transient states in parallel via MPI engines.

        Each worker explores with local indices starting from 0. After
        collection, rows are remapped to contiguous global indices by
        adding an offset equal to ``_next_state_index`` at merge time.
        """
        if not states_batch:
            return Ok(None)

        # Submit all exploration tasks with local (zero-based) indices
        futures = {}
        for state_idx in states_batch:
            kwargs = self._prepare_explore_kwargs(state_idx, start_index=0)
            futures[state_idx] = self.manager.basin_explore(**kwargs)

        # Collect results sequentially; remap local → global indices
        for state_idx, future in futures.items():
            try:
                rows = future.result()
            except Exception as exc:
                return self._transport_error("basin_explore", exc)
            if rows:
                offset = self._next_state_index
                for row in rows:
                    row["state_connexion"] += offset
                local_max = max(r["state_connexion"] for r in rows)
                self._next_state_index = local_max + 1
                self.connectivity_table.add_connectivity_batch(rows)

        return Ok(None)

    def is_new_state_batch(self, new_systems):
        """Check multiple systems for duplicates at once.

        Parameters
        ----------
        new_systems : dict[int, System]
            Mapping state_idx -> System for newly reconstructed states.

        Returns
        -------
        dict[int, int]
            Mapping state_idx -> existing_state_idx for duplicates,
            state_idx -> -1 for genuinely new states.

        """
        results = {}

        # Pre-compute fingerprints for new systems
        new_fingerprints = {}
        for idx, system in new_systems.items():
            new_fingerprints[idx] = self._compute_fingerprint(system.positions, system.cell, system.pbc)

        # Build fingerprint-filtered cKDTree only for candidate existing states
        # (pre-filter: only build trees for states whose fingerprint is close)
        existing_trees = {}
        for idx, state_data in self.states.items():
            if state_data.system is not None:
                if state_data.system.pbc is None or np.all(state_data.system.pbc):
                    box = np.diag(state_data.system.cell).tolist()
                    wrapped = self._wrap_positions(state_data.system.positions, state_data.system.cell)
                    existing_trees[idx] = cKDTree(wrapped, boxsize=box)
                else:
                    existing_trees[idx] = None  # fallback to manual comparison

        # Pre-compute fingerprint candidate sets for each new system vs existing states
        existing_fp_items = [
            (si, fp)
            for si, fp in self._state_fingerprints.items()
        ]

        fp_tol = self._fingerprint_tolerance()

        for new_idx, system in new_systems.items():
            match = -1
            fp_new = new_fingerprints[new_idx]

            # Fingerprint pre-filter against existing states
            if existing_fp_items:
                candidate_indices = []
                for si, fp in existing_fp_items:
                    if len(fp) == len(fp_new) and np.max(np.abs(fp - fp_new)) <= fp_tol:
                        candidate_indices.append(si)
            else:
                candidate_indices = list(existing_trees.keys())

            for existing_idx in candidate_indices:
                if existing_idx not in existing_trees:
                    continue
                tree = existing_trees[existing_idx]
                if tree is not None:
                    wrapped_query = self._wrap_positions(system.positions, system.cell)
                    distances, _ = tree.query(wrapped_query, k=1)
                    if np.max(distances) < 0.3:
                        match = existing_idx
                        break
                else:
                    state_data = self.states[existing_idx]
                    if self.are_structures_equivalent(system.positions, state_data.system.positions,
                                                      cell=system.cell, pbc=system.pbc):
                        match = existing_idx
                        break

            # Cross-check within this batch (two new states may be duplicates of each other)
            if match == -1:
                for other_idx in list(results.keys()):
                    if results[other_idx] != -1:
                        continue  # this one is already a duplicate itself
                    if other_idx in new_systems:
                        fp_other = new_fingerprints[other_idx]
                        # Fingerprint pre-filter within batch
                        if len(fp_other) == len(fp_new) and np.max(np.abs(fp_other - fp_new)) > fp_tol:
                            continue
                        if self.are_structures_equivalent(system.positions,
                                                          new_systems[other_idx].positions,
                                                          cell=system.cell, pbc=system.pbc):
                            match = other_idx
                            break

            results[new_idx] = match
        return results

    def construct_connexion_table_parallel(self):
        """Wavefront-parallel BFS: processes batches of states instead of one at a time.

        Phases per wavefront:
            A. Batch reconstruction (PSR + minimize)
            B. Batch deduplication
            C. Parallel exploration of new transient states
            D. Merge and update queue
        """
        import time

        strategy = self.config.basin.strategy
        n_workers = self.config.basin.n_workers

        t_start = time.perf_counter()
        n_explored = 0
        n_duplicates = 0
        n_absorbing = 0
        n_processed = 0
        prof = {"reconstruct": 0.0, "psr": 0.0, "minimize": 0.0,
                "dedup": 0.0, "ensure_state": 0.0, "explore": 0.0,
                "merge": 0.0, "other": 0.0}

        # Switch to session pool for basin reconstruction (parallel minimization)
        self.manager.use_local()

        max_states: int | None = self.config.basin.max_states
        t_last_log = t_start

        try:
            while len(self.states_to_explore) != 0:
                # Check max_states cap
                if max_states is not None and n_explored >= max_states:
                    logger.warning("[Basin] max_states=%d reached. Capping.", max_states)
                    result = self._cap_remaining_as_absorbing()
                    if not result.is_ok():
                        return result
                    break

                batch = list(self.states_to_explore)

                # Separate: states that need reconstruction vs state 0 (already exists)
                to_reconstruct = [s for s in batch if s not in self.states]
                already_exist = [s for s in batch if s in self.states]

                # ── Phase A: Batch reconstruction ──
                reconstructed = {}
                transition_info = {}
                if to_reconstruct:
                    t0 = time.perf_counter()
                    result = self._reconstruct_wavefront_batch(to_reconstruct)
                    prof["reconstruct"] += time.perf_counter() - t0
                    if not result.is_ok():
                        return result
                    reconstructed, transition_info, processed_count = result.ok_value()
                    n_processed += processed_count

                # ── Phase B: Batch deduplication ──
                # Always use batch dedup in the wavefront loop to catch intra-batch
                # duplicates.  Serial is_new_state() only checks against self.states
                # (which doesn't include other batch members), so duplicates within
                # the same batch go undetected — leading to exponential blowup.
                t0 = time.perf_counter()
                new_transient, duplicates_delta, absorbing_delta = self._register_wavefront_states(
                    to_reconstruct,
                    reconstructed,
                    transition_info,
                    prof,
                )
                prof["dedup"] += time.perf_counter() - t0
                n_duplicates += duplicates_delta
                n_absorbing += absorbing_delta

                # Also include pre-existing states that need exploration (e.g., state 0)
                for state_idx in already_exist:
                    if state_idx in self.states_to_explore:
                        new_transient.append(state_idx)

                # ── Phase C: Exploration via MPI engines ──
                if new_transient:
                    t0 = time.perf_counter()
                    result = self._explore_states_parallel(new_transient, n_workers=n_workers)
                    prof["explore"] += time.perf_counter() - t0
                    if not result.is_ok():
                        return result

                    # Mark explored
                    for state_idx in new_transient:
                        if state_idx in self.states_to_explore:
                            self.states_to_explore.remove(state_idx)
                        if state_idx not in self.explored_states:
                            self.explored_states.append(state_idx)
                        self.states[state_idx].release_heavy_objects()
                        n_explored += 1

                # ── Phase D: Update queue ──
                t0 = time.perf_counter()
                self.update_to_explore()
                prof["merge"] += time.perf_counter() - t0

                elapsed = time.perf_counter() - t_start
                logger.debug("[Basin] wavefront done | explored=%d | to_explore=%d | states=%d | dup=%d | abs=%d | %.1fs",
                             n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, elapsed)

                # Periodic progress log (every 30s)
                now = time.perf_counter()
                if now - t_last_log >= 30.0:
                    logger.info(
                        "[Basin] PROGRESS (wavefront): explored=%d | to_explore=%d | states=%d | duplicates=%d | absorbing=%d | %.0fs elapsed",
                        n_explored, len(self.states_to_explore), len(self.states), n_duplicates, n_absorbing, now - t_start,
                    )
                    t_last_log = now
        finally:
            self.manager.use_global()
        return self._finalize_exploration_run(
            t_start,
            prof,
            n_processed,
            n_duplicates,
            strategy_label=strategy,
        )
