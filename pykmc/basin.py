import pandas as pd
from pykmc import NeighborsList, AtomicEnvironment, PointSetRegistration, System, Engine
from .utils import geometry
import copy
import numpy as np
from scipy.spatial import cKDTree
from .config import Config
from .symmetries import unique_symmetries
from scipy.linalg import expm


# Set global options to always display the full DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)





class Basin() : 

    def __init__(self, reference_table):                                    
        self.connexion_table : pd.DataFrame = None
        self.states :  list[int] = []
        self.explored_states : list[int] = []
        self.states_to_explore : list[int] = None
        self.reference_table = reference_table
        self.state_system : list = None 
        self.state_environment : list[list[str|bytes]] = []
        self.state_neighbors_list: list = []
        self.applicable_events_df: list = []
        self.states_to_visit: list[int] = None
        self.visited_states: list[int] = []
        self.explored_environments: set[str|bytes] = None
      
    

    def detectin(self, selected_active_event_series, energy_threshold, index_event = None) : 

        dE_forward = selected_active_event_series["energy_barrier"]

        if dE_forward >= energy_threshold : 
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

        # Identify transient and absorbing states
        self.transient_states = self.explored_states
        self.absorbing_states = list(set(self.visited_states).difference(set(self.explored_states))) # à voir si y'a un problème pour un state déjà connu sous un autre nom

        # Apply fpta method
        if len(self.transient_states) > 0 and len(self.absorbing_states) > 0 :
            self.run_fpta()
        


    def initialize(self, system, config) :
        #Initialiser attribute avec info pour state de départ (0)
        self.temp = config.rateconstant.T   # Absolute temperature
        self.states = [0]
        self.state_system = [system]
        self.states_to_explore = []
        self.state_environment = []
        self.states_to_visit = []  
        self.visited_states = [0]   # ??
        self.connexion_table = pd.DataFrame(columns=['state', 
                                                     'state_connexion', 
                                                     'event_connexion',
                                                     'central_atom', 
                                                     'sym',
                                                     'transient'])    
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
        
    

    def update_connexion_table(self, state, config, transient = False) :

        #Initialize state_connexion
        if state == 0 :
            state_connexion = 1
        else :
            state_connexion = int(self.connexion_table.tail(1)['state_connexion'] + 1)     # find value of last state connexion in connexion_table
        
        if not transient :
            if not state_connexion in self.states_to_visit :
                self.states_to_visit.append(state_connexion)

        else :    

            # search for events applicable to the state
            self.applicable_events_df = self.get_applicable_generic_event(state)  


            # for all applicable events
            for idx_table, dfevent in self.applicable_events_df.iterrows() :

                energy_barrier = dfevent['energy_barrier']

                # search atoms on which we can apply the event
                l_atom_index = [atom_idx for atom_idx, atom_id in enumerate(self.get_atomic_environment(state)) if atom_id == dfevent['event_id']]

                is_transient = self.detectin(dfevent, config.basin.energy_thr, dfevent.name)

                for atom_idx in l_atom_index :
                    applicable_events = pd.DataFrame([{'state' : state, 
                                                    'state_connexion' : state_connexion, 
                                                    'event_connexion' : idx_table,
                                                    'central_atom' : atom_idx, 
                                                    'sym' : None,
                                                    'energy_barrier' : energy_barrier,
                                                    'transient' : is_transient}])
            
                    # symmetries
                    for idx in range(len(dfevent['sym_matrix'])) :
                        if idx == 0 :
                            applicable_events.at[idx, 'sym'] = idx 
                        else :
                            events_symmetries = pd.DataFrame([{'state' : state, 
                                                               'state_connexion' : state_connexion + idx, 
                                                               'event_connexion' : idx_table,
                                                               'central_atom' : atom_idx, 
                                                               'sym' : idx,
                                                               'energy_barrier' : energy_barrier,
                                                               'transient' : is_transient}])
                            
                            applicable_events = pd.concat([applicable_events, events_symmetries], ignore_index=True)



                    self.connexion_table = pd.concat([self.connexion_table, applicable_events], ignore_index=True)

                    # Check transient states
                    self.states_to_visit.append(state_connexion)

                    # Only visit the transient states
                    if is_transient == True :
                        self.states_to_explore.append(state_connexion) 

                    self.states.append(state_connexion)

                    state_connexion += 1
        
            self.explored_states.append(state)
        





    def apply_generic_event(self, config, reference_table, engine) :

        while len(self.states_to_visit) != 0 :

            #to_visit = next(s for s in self.states_to_visit if s not in self.visited_states)
            to_visit = self.states_to_visit[0]

            row = self.connexion_table.loc[self.connexion_table['state_connexion'] == to_visit]
            atom_index = row.iloc[0]['central_atom']
            from_state = row.iloc[0]['state']
            event_idx = row.iloc[0]['event_connexion']
            sym = row.iloc[0]['sym']

            
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
                    final_positions = ref_event.at['final_positions']
                    final_positions = geometry.transform_positions(final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)


                    # Apply symmetry matrix if sym != 0
                    if sym != 0 :
                        sym_matrices = ref_event.iloc[0]['sym_matrix']
                        sym_matrix = sym_matrices[sym]
                        final_positions = geometry.transform_positions(final_positions, sym_matrix)


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
                        self.connexion_table.loc[self.connexion_table['state_connexion'] == to_visit, 'transient'] = False


                    else :
                        # Register the new state
                        self.state_system.append(new_system)
                        self.update_state_environment(new_system, config, to_visit)
                        self.update_connexion_table(to_visit, config, to_visit in self.states_to_explore)     # Update the connexion_table only if the state is a transient state
                        


                    # Remove to_visit from states_to_visit
                    if to_visit in self.states_to_explore :
                        ind =  [i for i, e in enumerate (self.states_to_explore) if e == to_visit] [0]
                        self.states_to_explore.pop(ind)                                                            

                    # Update the states to check
                    self.visited_states.append(to_visit)
                    self.states_to_visit.pop(0)

                    self.debug_tables(to_visit)

                     
                     
                else : 
                    self.states_to_explore = []
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




# IMPLÉMENTER LA MÉTHODE FPTA

    def compute_jump_rate(self, energy_barrier) :
        jump_frequency = 10 ** 13 # idk
        boltzmann_constant = 1.380649 * 10 ** -23
        jump_rate = jump_frequency * np.exp( -energy_barrier / (boltzmann_constant * self.temp))
        return jump_rate

    

    def build_rate_matrix(self) :
        self.connexion_table['jump_rate'] = self.connexion_table['energy_barrier'].apply(self.compute_jump_rate)

        self.transition_matrix = None
        self.occupation_probabilities = None

        print("Transient states:" , len(self.transient_states))
        print("Absorbing states:" , len(self.absorbing_states))


        # Construction de la matrice M pour tous les états (transitoires + absorbants)
        all_states = self.transient_states + self.absorbing_states
        n_total = len(all_states)
        
        # Mapping état -> index
        state_to_idx = {state: i for i, state in enumerate(all_states)}
        
        # Initialisation de la matrice M
        M = np.zeros((n_total, n_total))


        # Remplissage de la matrice selon l'équation (A4)
        for _, row in self.connexion_table.iterrows():
            state_i = row['state']
            state_j = row['state_connexion']
            rate = row['jump_rate']
            
            if state_i in state_to_idx and state_j in state_to_idx:
                i = state_to_idx[state_i]
                j = state_to_idx[state_j]
                
                # Mij = -Rj->i si i ≠ j
                if i != j:
                    M[j, i] = -rate


        # Diagonale : Mii = ∑k Ri->k 
        for i in range(n_total):
            # Calculer la somme des taux sortants pour l'état i
            outgoing_rates = 0
            state_i = all_states[i]
            
            # Parcourir la connexion_table pour trouver tous les taux sortants
            for _, row in self.connexion_table.iterrows():
                if row['state'] == state_i :
                    outgoing_rates += row['jump_rate']
            
            M[i, i] = outgoing_rates
        
        self.M_matrix = M
        return M            
    



    def build_reduced_matrix(self): 
        # Construit la matrice M réduite en considérant tous les états absorbants comme un seul
        n_transient = len(self.transient_states)
        M_reduced = np.zeros((n_transient + 1, n_transient + 1))
        
        # Copier la partie transitoire
        M_reduced[:n_transient, :n_transient] = self.M_matrix[:n_transient, :n_transient]
        
        # Sommer les transitions vers les états absorbants
        for i in range(n_transient):
            state_i = self.transient_states[i]
            rate_to_absorbing = 0
            
            # Sommer tous les taux vers les états absorbants
            for _, row in self.connexion_table.iterrows():
                if row['state'] == state_i and row['state_connexion'] in self.absorbing_states:
                    rate_to_absorbing += row['jump_rate']



            # Transition vers l'état absorbant fusionné (colonne n_transient)
            M_reduced[n_transient, i] = -rate_to_absorbing  # M[j,i] = -R(i->j)
            
            # Recalculer la diagonale (somme des taux sortants de l'état i)
            total_absorbing_rate = rate_to_absorbing  # vers états absorbants
            
            # Ajouter les taux vers autres états transitoires
            for j in range(n_transient):
                if i != j:
                    # Trouver le taux i -> j
                    for _, row in self.connexion_table.iterrows():
                        if (row['state'] == state_i and 
                            row['state_connexion'] == self.transient_states[j]):
                            total_absorbing_rate += row['jump_rate']
                            break
            
            M_reduced[i, i] = total_absorbing_rate   

        return M_reduced




    def run_fpta_step(self, current_state):
        # Exécute une étape complète de FPTA
        initial_state_idx = self.transient_states.index(current_state)
        n_transient = len(self.transient_states)
        
        # Construire la matrice de taux
        self.build_rate_matrix()
        
        # Matrice réduite (états absorbants fusionnés)
        M_reduced = self.build_reduced_matrix()

        # Vecteur initial P̄(0)
        initial_vector = np.zeros(n_transient + 1)
        initial_vector[initial_state_idx] = 1.0
        
        # Nombre aléatoire r pour probabilité de sortie
        random_r = np.random.random()

        # Temps initial t₀
        initial_time = 1.0 / np.sum(np.diag(self.M_matrix))  # basé sur ce qu'ils ont utilisé dans l'article
        
        # Bissection pour trouver texit
        t_min, t_max = 0.0, initial_time
        tolerance = 1e-5   # basé sur ce qu'ils ont utilisé dans l'article


        # Étendre la recherche si le chiffre random r est plus grand que la probabilité d'absorption
        while True :
            prob_vector = expm(-t_max * M_reduced) @ initial_vector
            prob_absorbing = prob_vector[-1]
            
            if prob_absorbing > random_r:
                break
            t_max *= 2
        
        # Bissection
        while True :
            t_mid = (t_min + t_max) / 2
            prob_vector = expm(-t_mid * M_reduced) @ initial_vector
            prob_absorbing = prob_vector[-1]
            
            if abs(t_max - t_min) / ((t_max + t_min) / 2) < tolerance:
                break
            
            if prob_absorbing < random_r:
                t_min = t_mid
            else:
                t_max = t_mid
        
        exit_time = t_mid
        print("Temps de sortie texit =" , exit_time)
        

        # Probabilités individuelles des états absorbants
        n_total = len(self.transient_states) + len(self.absorbing_states)
        initial_vector_full = np.zeros(n_total)
        initial_vector_full[initial_state_idx] = 1.0
        
        prob_vector_full = expm(-exit_time * self.M_matrix) @ initial_vector_full
        absorbing_probs = prob_vector_full[n_transient:]
        

        # Normalisation  ??? 
        total_absorbing_prob = np.sum(absorbing_probs)
        if total_absorbing_prob > 0:
            absorbing_probs = absorbing_probs / total_absorbing_prob
        
        print("Probabilités des états absorbants:" , absorbing_probs)
        

        # Sélection de l'état final
        random_s = np.random.random()
        
        cumulative_prob = 0
        selected_state = None
        for i, prob in enumerate(absorbing_probs):
            cumulative_prob += prob
            if random_s <= cumulative_prob:
                selected_state = self.absorbing_states[i]
                break
        
        if selected_state is None:
            selected_state = self.absorbing_states[-1]
        
        print("État sélectionné:" , selected_state)
        
        return selected_state, exit_time    
    



    # Dans l'article il est écrit qu'un state doit être visité 10 fois avant qu'on lui applique fpta. Est-ce que c'est applicable à nous?? Ou est-ce qu'on
    # calcule seulement le temps de sortie selon l'état qui nous à amené dans le bassin soit l'état 0??




    def run_fpta(self) :

        total_exit_time = 0
        simulation_results = []

        # Execute 10 fpta steps
        for step in range(10) :
            next_state, exit_time = self.run_fpta_step(0)   # Je sais pas on est supposé appliquer ftpa à partir de quel state???
            total_exit_time += exit_time

            # Simulation results
            simulation_results.append(next_state)

        print(simulation_results, total_exit_time)
        
        return simulation_results, total_exit_time