class Basin() : 


    def detectin(self, selected_active_event_series) : 

        dE_forward = selected_active_event_series['energy_barrier']
        dE_backward = selected_active_event_series['backward_energy_barrier']

        if dE_forward > 0.15  and dE_backward > 0.15 : 
            print ('Basin not detected')
            return False
        
        else : #event is symmetrical and energy barriers are low
            print('Basin detected')
            return True







    def execute(self) : 
        pass








    def reset() : 
        pass 