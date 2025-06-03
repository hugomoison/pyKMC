from kmc import KMC



class Basin() : 


    def detectin(selected_active_event_series) : 

        dE_forward = selected_active_event_series['energy_barrier']
        dE_backward = selected_active_event_series['backward_energy_barrier']

        if dE_forward > 0.15  and dE_backward > 0.15 :
            print ('Basin not detected')
            return False
        
        else :
            print('Basin detected')
            return True












    def execute() : 
        pass 

    def reset() : 
        pass 