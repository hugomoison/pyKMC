from pykmc import System, NeighborsList, AtomicEnvironment, ReferenceEventTable, PointSetRegistration, Config
from pykmc import geometry
from ase import Atoms
from ase.io import write 

config = Config.from_file('input.in')

system = System.create_from_file('initial_config.xyz') 
neighbors_list = NeighborsList(system, config) 
atomic_environment = AtomicEnvironment(config, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut']) 
reference_table = ReferenceEventTable(config)  #TABLE AVEC SEULEMENT UN EVENT
reference_table_df = reference_table.table.iloc[0]




system_initial_positions = system.positions.copy()
event_id = reference_table_df['event_id']

#Cherche un atom qui a cet ID 

atom_index = [i for i, e in enumerate(atomic_environment.atomic_environment_list) if e == event_id][4]  

#ON VEUT APPLIQUER EVENEMENET GERENIQUE A ATOM INDEX 

#1. Point Set Registration entre Environment atomique de atom_index et reference_table.table.loc[0].at['initial_positions'] --> Utiliser la class PointSetRegistration

#Need to go to final point applying PSR : 
psr_output = PointSetRegistration(config, system, reference_table_df, neighbors_list, 0, atom_index).run()


if psr_output.is_ok():
    psr_output = psr_output.ok_value()

    #2. Checker si Point Set Registration est ok  -> Si dh < 0.3
    if psr_output.matching_score < 0.3 :


        #3. Appliquer le Point Set Registration a reference_table.table.loc[0].at['final_positions'] --> donne new_positions  (utiliser np.matmul ou operateur python @) 
        #Apply PSR to generic event
        final_positions = geometry.transform_positions(reference_table_df.at['final_positions'], psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)

        #Get atomic environment atoms
        neighbors = neighbors_list.get_neighbors('rcut', atom_index)



        #4. Changer les positions de l'environment atomique de l'atom atom_index avec celles de new positions. --> utilise system.update_position(new_positions, atom_idx =[list atom dans l'environment atomique de l'atom central atom_index])
        #Move system do final positions
        system.update_positions(final_positions, atom_idx = neighbors)
       

        ### Pourquoi on met atom_idx = neighbours??  self.system.update_positions(saddle_positions, atom_idx = neighbors) ### 
   
    else : 
        print("PSR found a match but matching score is above acceptance threshold")


else:
    print("Registration failed:", psr_output.err_value())



#5. Ecrire dans un fichier les configurations pour voir si c'est ok. -> utiliser ase.io.write
#atoms1 = Atoms(symbols = system.types, positions = system_initial_positions, cell = system.cell, pbc = True) 
#atoms2 = Atoms(symbols = system.types, positions = system.positions, cell = system.cell, pbc = True)
#traj = [atoms1, atoms2] 
#write('psrtest.xyz', traj)

atoms1 = Atoms(symbols = system.types, positions = system_initial_positions, cell = system.cell, pbc = True) 
atoms2 = Atoms(symbols = system.types, positions = system.positions, cell = system.cell, pbc = True)
traj = [atoms1, atoms2] 
write('psrtest.xyz', traj)