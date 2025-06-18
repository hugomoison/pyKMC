import pandas as pd
from pykmc import NeighborsList, AtomicEnvironment, PointSetRegistration, System



class Basin() : 

    def __init__(self, reference_table):                                    
        self.connexion_table : pd.DataFrame = None
        self.states :  list[int] = []
        self.visited_states : list[int] = []
        self.states_to_visit : list[int] = None
        self.reference_table = reference_table
        self.state_system : list = None 
        self.state_environment : list[list[str|bytes]] = []



    def detectin(self, selected_active_event_series, energy_threshold) : 

        dE_forward = selected_active_event_series['energy_barrier']
        dE_backward = selected_active_event_series['backward_energy_barrier']

        if dE_forward > energy_threshold  and dE_backward > energy_threshold  : 
            print ('Basin not detected')
            return False
        
        else : #event is symmetrical and energy barriers are low
            print('Basin detected')
            return True




    def execute(self, initial_system, config) : 

        self.initialize(initial_system, config)

        # ---> Implémenter quelque chose de similaire au test à partir du point #1










    def initialize(self, system, config) :
        #Initialiser attribute avec info pour state de départ (0)
        self.states = [0]
        self.states_to_visit = [0]          # ??? Est-ce que on peut l'initilaiser comme ça ici?
        self.state_system = [system]
        self.state_environment = []
        self.connexion_table = pd.DataFrame(columns=['state', 
                                                     'state_connexion', 
                                                     'event_connexion',
                                                     'central_atom', 
                                                     'transition', ])    
        self.update(0, config )




    def update(self, state , config) :
        self.update_state_environment(self.state_system[state], config)   
        self.get_applicable_generic_event(state)
        self.update_connexion_table(state, config)
        self.debug_tables(state)



    def update_state_environment(self, system, config) :  
        neighbors_list = NeighborsList(system, config)  
        atomic_environment = AtomicEnvironment(config, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'])
        self.state_environment.append(atomic_environment.atomic_environment_list.copy())
    



    def get_applicable_generic_event(self, state) :
        self.atomic_events_id = self.get_atomic_environment(state)
        df = self.reference_table.table
        applicable_generic_events = df[df['event_id'].isin(self.atomic_events_id)]
        return applicable_generic_events    # returns sub DataFrame of reference_table with events that have an ID in AtomicEnvironment_list


    
    def get_atomic_environment(self, state) :    
        return self.state_environment[state]     # returns atomic environment associated with the state
        
    

    def update_connexion_table(self, state, config) :
        
        # search for events applicable to the state
        applicable_events_df = self.get_applicable_generic_event(state)  

        ### applicable_event_ids = set(applicable_events_df['event_id'])

        #Initialize state_connexion
        if state == 0 :
            state_connexion = 1
        else :
            state_connexion = self.connexion_table.tail(1)['state_connexion'] + 1     # find value of last state connexion in connexion_table

        # for all applicable events
        for idx_table, dfevent in applicable_events_df.iterrows() :
            # search atoms on which we can apply the event
            l_atom_index = [atom_idx for atom_idx, atom_id in enumerate(self.get_atomic_environment(state)) if atom_id == dfevent['event_id']]

            is_transition = self.detectin(dfevent, config['Basin']['energy_thr'])

            for atom_idx in l_atom_index :
                applicable_events = pd.DataFrame([{'state' : state, 
                                                   'state_connexion' : state_connexion, 
                                                   'event_connexion' : idx_table,
                                                   'central_atom' : atom_idx, 
                                                   'transition' : is_transition}])
        

                
                self.connexion_table = pd.concat([self.connexion_table, applicable_events], ignore_index=True)


                if is_transition == True :
                    self.states_to_visit.append(state_connexion)

                state_connexion += 1


                self.states.append(state_connexion)


       
        self.visited_states.append(state)
        





    def apply_generic_event(self, config, initial_system, state, idx_selected_event, reference_table) :

        while (set(self.visited_states) != set(self.states_to_visit)) :
            to_visit = list(set(self.states_to_visit).difference(set(self.visited_states))) [0]

            #Need to go to final point applying PSR : 
            psr_output = PointSetRegistration(config, initial_system,  neighbors_list, 0, atom_index).run()



            self.visited_states.append(to_visit)
            print('roar')
         #   self.apply_generic_event(config, self.reference_table) 





    #    new_positions = reference_table.table.loc[idx_selected_event].at['final_positions']
     #   self.system.update_positions(new_positions) 







    def debug_tables(self, state):
        print("\n=== Connexion Table ===")
        print(self.connexion_table)

        print("\n=== Applicable Generic Events ===")
        print(self.get_applicable_generic_event(state))











    def reset() : 
        pass 