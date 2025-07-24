import pandas as pd
from pykmc import NeighborsList, AtomicEnvironment, PointSetRegistration, System, Engine
from .utils import geometry
import copy
import numpy as np
from scipy.spatial import cKDTree


# Set global options to always display the full DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)





class Basin() : 

    def __init__(self, reference_table):                                    
        self.connexion_table : pd.DataFrame = None
        self.states :  list[int] = []
        self.visited_states : list[int] = []
        self.states_to_visit : list[int] = None
        self.reference_table = reference_table
        self.state_system : list = None 
        self.state_environment : list[list[str|bytes]] = []
        self.state_neighbors_list: list = []
        self.applicable_events_df: list = []
        self.states_to_check: list[int] = None
        self.checked_states: list[int] = []
        self.explored_environments: set[str|bytes] = None
      
    

    def detectin(self, selected_active_event_series, energy_threshold, index_event = None) : 

        dE_forward = selected_active_event_series["energy_barrier"]

        if dE_forward > energy_threshold : 
            print ('Basin not detected')
            return False
        
        else : # Forward energy barrier is low 

            # Get row of forward event
            if index_event != None :
                num_reference_event = index_event

            else :
                num_reference_event = selected_active_event_series["num_reference_event"].item()

            forward_event_row = self.reference_table.table.iloc[num_reference_event]

            forward_id_final = forward_event_row["id_final"]   
            forward_id_initial = forward_event_row["event_id"]


            # Get row of the potential backward event
            backward_event_row =  self.reference_table.table[self.reference_table.table["event_id"] == forward_id_final]

            if not backward_event_row.empty and backward_event_row.iloc[0]["id_final"] == forward_id_initial :
                dE_backward = backward_event_row.iloc[0]["energy_barrier"]


            if dE_backward < energy_threshold :
                print("Basin detected")
                return True
            else :
                return False




    def execute(self, initial_system, config, reference_table, engine, explored_environments) : 
        self.explored_environments = explored_environments
        self.initialize(initial_system, config)
        self.apply_generic_event(config, reference_table, engine)




    def initialize(self, system, config) :
        #Initialiser attribute avec info pour state de départ (0)
        self.states = [0]
        self.state_system = [system]
        self.states_to_visit = []
        self.state_environment = []
        self.states_to_check = []  
        self.checked_states = [0]   # ??
        self.connexion_table = pd.DataFrame(columns=['state', 
                                                     'state_connexion', 
                                                     'event_connexion',
                                                     'central_atom', 
                                                     'transition', ])    
        self.update(0, config)




    def update(self, state , config) :
        self.update_state_environment(self.state_system[state], config, state)   
        self.get_applicable_generic_event(state)
        self.update_connexion_table(state, config, True)
        self.debug_tables(state)




    def update_state_environment(self, system, config, state, state_index = None) :  
        neighbors_list = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)  
        atomic_environment = AtomicEnvironment(config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], config.atomicenvironment.neighbors_add)
        if state_index is None:

            self.state_environment.append(atomic_environment.atomic_environment_list.copy())
            self.state_neighbors_list.append(neighbors_list)
        else:
            self.state_environment[state_index] = atomic_environment.atomic_environment_list.copy()
            self.state_neighbors_list[state_index] = neighbors_list

        return neighbors_list




    def get_applicable_generic_event(self, state) :
        self.atomic_events_id = self.get_atomic_environment(state)
        df = self.reference_table.table
        applicable_generic_events = df[df['event_id'].isin(self.atomic_events_id)]
        return applicable_generic_events    # returns sub DataFrame of reference_table with events that have an ID in AtomicEnvironment_list



    
    def get_atomic_environment(self, state) :    
        return self.state_environment[state]     # returns atomic environment associated with the state
        
    

    def update_connexion_table(self, state, config, transition = False) :

        #Initialize state_connexion
        if state == 0 :
            state_connexion = 1
        else :
            state_connexion = int(self.connexion_table.tail(1)['state_connexion'] + 1)     # find value of last state connexion in connexion_table
        
        if not transition :
            if not state_connexion in self.states_to_check :
                self.states_to_check.append(state_connexion)

        else :    

            # search for events applicable to the state
            self.applicable_events_df = self.get_applicable_generic_event(state)  


            # for all applicable events
            for idx_table, dfevent in self.applicable_events_df.iterrows() :
                # search atoms on which we can apply the event
                l_atom_index = [atom_idx for atom_idx, atom_id in enumerate(self.get_atomic_environment(state)) if atom_id == dfevent['event_id']]

                is_transition = self.detectin(dfevent, config.basin.energy_thr, dfevent.name)

                for atom_idx in l_atom_index :
                    applicable_events = pd.DataFrame([{'state' : state, 
                                                    'state_connexion' : state_connexion, 
                                                    'event_connexion' : idx_table,
                                                    'central_atom' : atom_idx, 
                                                    'transition' : is_transition}])
            

                    self.connexion_table = pd.concat([self.connexion_table, applicable_events], ignore_index=True)

                    # Check transition states
                    self.states_to_check.append(state_connexion)

                    # Only visit the transition states
                    if is_transition == True :
                        self.states_to_visit.append(state_connexion) 

                    self.states.append(state_connexion)

                    state_connexion += 1
        
            self.visited_states.append(state)
        





    def apply_generic_event(self, config, reference_table, engine) :

        while len(self.states_to_check) != 0 :

            #to_visit = next(s for s in self.states_to_check if s not in self.checked_states)
            to_visit = self.states_to_check[0]

            row = self.connexion_table.loc[self.connexion_table['state_connexion'] == to_visit]
            atom_index = row.iloc[0]['central_atom']
            from_state = row.iloc[0]['state']
            event_idx = row.iloc[0]['event_connexion']

            
            # Get generic event from reference_table
            ref_event = reference_table.table.loc[event_idx]


            # Get neighbors and update state_environment
            self.update_state_environment(self.state_system[from_state], config, to_visit, state_index = from_state)     #?
            neighbors = self.state_neighbors_list[from_state].get_neighbors('rcut', atom_index)
            

            # Go to final point applying PSR
            psr_output = PointSetRegistration(config, self.state_system[from_state], ref_event, self.state_neighbors_list[from_state], atom_index).match()
            verification_system = self.state_system[from_state]

            if psr_output.is_ok():
                psr_output = psr_output.ok_value()

                # Check if PointSetRegistration finds a match
                if psr_output.matching_score < 0.3 :
 
                    # Apply PSR to generic event
                    final_positions = reference_table.table.loc[event_idx].at['final_positions']
                    final_positions = geometry.transform_positions(final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)


                    # Move system do final positions
                    new_system = copy.deepcopy(self.state_system[from_state])
                    new_system.update_positions(final_positions, atom_idx = neighbors)


                    # Minimize after moving the system
                    new_positions, total_energy = engine.minimize(new_system)
                    new_system.update_positions(new_positions)

                    # Update state_environment for new state
                    self.update_state_environment(new_system, config, to_visit)


                    # Check if the state added in state_connexion has already been given another number in connexion_table
                    existing_state_index = self.find_existing_state(new_system)


                    if existing_state_index is not None :
                        print("State", to_visit, "is already known as state", existing_state_index)

                        # Update connexion_table to reflect this match
                        self.connexion_table.loc[self.connexion_table["state_connexion"] == to_visit, "state_connexion"] = existing_state_index

                        self.state_system.append(self.state_system[existing_state_index])
                        self.update_state_environment(self.state_system[existing_state_index], config, existing_state_index)


                    elif set(self.state_environment[to_visit]).difference(self.explored_environments) != set() :
                        hihi = list(set(self.state_environment[to_visit]).difference(self.explored_environments))
                        self.connexion_table.loc[self.connexion_table['state_connexion'] == to_visit, 'transition'] = False


                    else :
                        # Register the new state
                        self.state_system.append(new_system)
                        self.update_state_environment(new_system, config, to_visit)
                        self.update_connexion_table(to_visit, config, to_visit in self.states_to_visit)     # Update the connexion_table only if the state is a transient state
                        


                    # Remove to_visit from states_to_visit
                    if to_visit in self.states_to_visit :
                        ind =  [i for i, e in enumerate (self.states_to_visit) if e == to_visit] [0]
                        self.states_to_visit.pop(ind)                                                            

                    # Update the states to check
                    self.checked_states.append(to_visit)
                    self.states_to_check.pop(0)

                    self.debug_tables(to_visit)

                     


                else : 
                    self.states_to_visit = []
                    print("PSR found a match but matching score is above acceptance threshold")


            else:
                print("Registration failed:", psr_output.err_value())
               
        


    def are_structures_equivalent(self, pos1, pos2, system, tol = 0.1):
        """
        Compare two sets of atomic positions, ignoring atom order.
        """
        if len(pos1) != len(pos2):
            return False
        box = [system.cell[0][0], system.cell[1][1], system.cell[2][2]]
        tree1 = cKDTree(pos1, boxsize = box)    
        tree2 = cKDTree(pos2, boxsize = box)

        # Pour chaque point de pos1, trouver son plus proche voisin dans pos2
        distances1, _ = tree2.query(pos1, k=1)
        distances2, _ = tree1.query(pos2, k=1)

        max_dist1 = np.max(distances1)
        max_dist2 = np.max(distances2)

        return max(max_dist1, max_dist2) < tol




    def find_existing_state(self, system, tolerance = 0.1):      
        candidate_pos = system.positions                         

        for i, existing_system in enumerate(self.state_system):
            existing_pos = existing_system.positions

            if self.are_structures_equivalent(candidate_pos, existing_pos, system, tol = tolerance):
                print("✅ Match found with state", i)
                return i

        return None
        



    def debug_tables(self, state):
        print("\n=== Connexion Table ===")
        print(self.connexion_table)

       # print("\n=== Applicable Generic Events ===")
       # print(self.get_applicable_generic_event(state))





    def reset() : 
        pass 