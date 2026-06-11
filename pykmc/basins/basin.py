from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import System, Config, NeighborsList, AtomicEnvironment, ReferenceEventTable, PointSetRegistration, check_match, Reconstruction
from typing import Optional
from ..utils import geometry
from ..rate_constant import compute_rate_Eyring
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, BasinOutput

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
                self.environment = AtomicEnvironment(config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], config.atomicenvironment.neighbors_add, types=types, coordination_threshold=config.atomicenvironment.coordination_threshold)


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
        self.known_environments = known_environments 
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}

    def detection(self, params) -> bool : 
        """Utility method."""
        return self.detector.detection(**params) 
    
    def execute(self, system) : 
        """ 
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        #initialize the basin
        self._initialize(system)
        #explore the basin
        result = self.construct_connexion_table()
        if not result.is_ok() : 
            return result
        #reorder states index 
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}
        #Refine absorbing states
        self.manager.use_local()
        result =self.refine_absorbing(system)
        if not result.is_ok() : 
            return result
        #apply selector algorithm to find t_exit and exit_state
        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok() : 
            return result
        #Construct output KMC needs 
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=exit_state)
        #Ensure from_state is state are full 
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
        self.states_to_explore = [0] 
        self.explored_states = [] 
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = FPTASelector()
        new_system = System(positions=system.positions.copy(), types=system.types.copy(), cell=system.cell.copy(), pbc=system.pbc.copy(), index=np.arange(len(system.types)))
        self._add_state(state_index=0, system=new_system)  #add current state 0 to self.states


    def construct_connexion_table(self) : 
        """ 
        explore the basin and construct the connextion table
        """
        #Loop over state to explore 
        while len(self.states_to_explore) != 0 :
            #next state to explore : 
            to_explore = self.states_to_explore[0]

            if to_explore not in self.states : #always true except at the start (to_explore = 0) 
                #We need to create the state 
                    #find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)

                    #Create new system by applying (reconstruction) the generic event to the from_state
                result = self.system_from_state(from_state, event_idx, central_atom, sym_idx) 
                if not result.is_ok() : 
                    return result
                new_system = result.ok_value()

                    #Check if it is a new_system or already in states 
                is_new_state = self.is_new_state(new_system) 
                if is_new_state != -1 : #It already exists 
                    #update table
                    self.connectivity_table.change_state_index(current_index=to_explore, new_index=is_new_state)
                    self.explored_states.append(to_explore)
                    self.states_to_explore.remove(to_explore)

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue #Skip the rest

                #add state
                self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                #ENSURE FULL STATE TO EXPLORE 
                self.states[to_explore].ensure_full_state(self.config)
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

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()


                    continue #We dont explore/skip the rest

                #Release heavy objet memory
                self.states[from_state].release_heavy_objects()

            


            #Explore state 
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()

            #Ensure full state to explore 
            self.states[to_explore].ensure_full_state(self.config)
            self.explorer.explore(state=self.states[to_explore], state_index=self.current_state, start_index=last_state_connectivity)
            
            #to_explore has been explored : 
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            #Merge state connectivity table to basin connectivity table 
            self.connectivity_table.merge(self.explorer.connectivity_table)
            #Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            #Clean heaby state object : 
            self.states[to_explore].release_heavy_objects()
            
        return Ok(None)

    def select_event(self) : 
        """ 
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        convinient method
        """
        pass

    def get_last_state_index(self) : 
        if self.current_state == 0 : #connextion table is empty
            new_state_connexion = 1 
        else : #last state connexion +1
            new_state_connexion = int(self.connectivity_table.get_table()['state_connexion'].iloc[-1]+1)
        return new_state_connexion
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()['state']).union(set(self.connectivity_table.get_table()['state_connexion']))
        self.states_to_explore =  list(unique_states.difference(set(self.explored_states)))


    def system_from_state(self, from_state, event_idx, central_atom, sym_idx) : 
        """ Reconstruct the generic event to generate new state from state
        """

        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx] #event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
#        ref_event = self.reference_table.table.iloc[event_idx].copy()

        #supposed_initial_positions = ref_event["initial_positions"].copy()
        #supposed_final_positions = ref_event["final_positions"].copy()
        #saddle_positions = ref_event['saddle_positions'].copy()

        supposed_initial_positions = np.array(ref_event["initial_positions"], copy=True)
        supposed_final_positions = np.array(ref_event["final_positions"], copy=True)
        saddle_positions = np.array(ref_event['saddle_positions'], copy=True)

        #Apply the generic event to the current state 

        #ENSURE FULL STATE FOR FROM STATE 
        self.states[from_state].ensure_full_state(self.config)

            #We start from the from_state
        new_system = System(positions=self.states[from_state].system.positions.copy(), types=self.states[from_state].system.types, cell=self.states[from_state].system.cell, pbc=self.states[from_state].system.pbc, index=np.arange(len(self.states[from_state].system.types)))
        #new_system = copy.deepcopy(self.states[from_state].system)

            #Apply PSR between event initial position and environment positions of the central_atoms
        result = PointSetRegistration(self.config, new_system, ref_event , self.states[from_state].neighbors_list, central_atom).match()
        if not result.is_ok(): #PSR Err
            return result
            # Check if PointSetRegistration match is valid 
        result = check_match(result, self.config.psr.matching_score_thr)
        if not result.is_ok() : #PSR matching score not valid : 
            return result
        else : 
            psr_output = result.ok_value() #get psr results
            
        # Apply PSR to generic event to move 
            
        # Apply symmetry matrix if sym != 0
        if sym_idx != 0 :
            sym_matrices = ref_event['sym_matrix']
            sym_matrix = sym_matrices[sym_idx]
            supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            supposed_final_positions = geometry.transform_positions(supposed_final_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
        supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        supposed_final_positions= geometry.transform_positions(supposed_final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)

        # Move system do saddle positions
        neighbors = self.states[from_state].neighbors_list.get_neighbors('rcut', central_atom)

        if self.config.basin.style == "global" : 
            new_system.update_positions(supposed_final_positions, atom_idx=neighbors)
            min2_pos, _ = self.manager.global_minimize_with_results(self.config, positions=new_system.positions.copy())
            new_system.update_positions(min2_pos)

        elif self.config.basin.style == "global/reconstruction" : 
            new_system.update_positions(saddle_positions, atom_idx = neighbors)

            #Reconstruct the event
            #future = self.manager.minimize_with_results(self.config, positions=new_system.positions)
            #min_pos, _ = future.result()

            result = Reconstruction(self.config, self.manager).reconstruct(supposed_initial_positions, supposed_final_positions, new_system.positions, new_system.cell, self.config.psr.matching_score_thr, neighbors, pbc=new_system.pbc)
            if not result.is_ok() :
                return result
            new_system.update_positions(result.ok_value().min2_positions)
        
        else : 
            raise ValueError(f"Unknown {self.config.basin.style} style parameter.")

        return Ok(new_system)

    def refine_absorbing(self, system) :
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        #compute the energy of the state 
        #for all row in connectivity table where we need to refine
        futures_context = {} #idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.iterrows() : 
            if row['transient']  == False : #need to refine
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
                saddle_positions = ref_event['saddle_positions'].copy()
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
                    sym_matrices = ref_event['sym_matrix']
                    sym_matrix = sym_matrices[row["sym"]]
                    saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][row["sym"]])
                saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors('rcut', row["central_atom"])

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
            idx_state = self.connectivity_table.df.loc[idx].at['state_connexion']
            central_atom = self.connectivity_table.df.loc[idx].at['central_atom']
            #self.absorbing_saddle_positions[idx_state] = result.ok_value().saddle_positions[self.states[idx_state].neighbors_list.get_neighbors("rcut", central_atom)]
            self.absorbing_saddle_positions[idx_state] = result_sad.ok_value().saddle_positions[ctx["neighbors"]]
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        return Ok(None)


    def is_new_state(self, system) : 
        #Loop over all other system in self.states to see if system is already known

        for state_index, state_data in self.states.items():
            are_equivalent = self.are_structures_equivalent(system.positions, state_data.system.positions, cell = system.cell, pbc=system.pbc)
            if are_equivalent : 
                return state_index
        return -1 


    def are_structures_equivalent(self, pos1, pos2, cell, pbc=None, tol=0.3):

        if len(pos1) != len(pos2):
            return False

        if pbc is None or np.all(pbc):
            # Fully periodic: use boxsize (existing fast path)
            box = np.diag(cell).tolist()
            tree2 = cKDTree(pos2, boxsize=box)
            distances, _ = tree2.query(pos1, k=1)
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

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False, full=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing 
        neighbors_list  = []
        atomic_environment = []

        if full == True : 
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)  
            types = system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add, types=types, coordination_threshold=self.config.atomicenvironment.coordination_threshold)
        else : 
            neighbors_list = None 
            atomic_environment = None 
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state
