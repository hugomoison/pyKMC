import random
from lammps import lammps
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from subprocess import run
import pypARTn2
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import pynauty
from .atomic_environment import make_graph 
import logging
from .config import Parameters
import math as m
from .utilities import initialize_default_lammps


class EventSearch() : 
    """
    Define and run event search procedure    

    Attributes
    ----------
    system : System Object
        the current system
    search_style : str
        style use for the event search, can be 'pARTn', 'dimer'
    search_params : dict 
        parameter needed by the style used to perform the event search
    atomenv_params : dict 
        dictionaty of radius parameters defining the environment 
    potential : dict of str: str
        commands to define the potential used by the program defined by minimization_style
    reconstruction : boolean, optional 
        if we use events from previous searches and reconstruct them, by default 'True'
    dimension : int, optional
        dimension of the system, by default 3
    nprocs : int, optional
        number of procs available, by default 1
    backend : str, optional [deprecated]
        parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

    Methods 
    ------- 
    run() 
        run the event search 
    search_with_reconstruction() 
        event search procedure when `reconstruction == True`
    search_without_reconstruction()
        event search procedure when `reconstruction == False`
    new_environment() 
        find list of environment ID of the current system that are not in the catalog
    compute_rate_Eyring(dE) 
        compute the rate constant 
    add_event_with_reconstruction(dfevent_forward, dfevent_backward)
        procedure to add event to the catalog when `reconstruction == True`
    add_event_without_reconstruction(dfevent)
        procedure to add event to the catalog when `reconstruction == False`
    event_series_with_reconstruction(min1_positions, saddle_positions, min2_positions, index_move, dE_forward, dE_backward)
        create event pandas.Series from event search when `reconstruction == True`
    event_series_without_reconstruction(atom_index, final_positions, dE)
        create event pandas.Series from event search when `reconstruction == False`
    pARTn_search(atom_index)
        run an event search using pARTn 
    
    """

    def __init__(self, system, search_style, search_params, atomenv_params, potential, reconstruction, dimension, nprocs, backend) -> None:
        self.system = system 
        self.search_style = search_style
        self.search_params = search_params 
        self.atomenv_params = atomenv_params
        self.potential = potential
        self.reconstruction = reconstruction
        self.dimension = dimension 
        self.nprocs = nprocs 
        self.backend = backend


    def run(self) : 
        """ 
        Execute new event searches 
        """

        #======================================#
        #Check if we want reconstruction or not# 
        #======================================#
        match self.reconstruction : 
            case True : 
                self.search_with_reconstruction()
            case False : 
                self.search_without_reconstruction()
            case _: 
                raise Exception("Wrong reconstruction value in 'Control', must be True or False")


    def search_with_reconstruction(self) : 
        """
        Event search procedure when `reconstruction == True` 
        For each new topology ID, perform `nsearch` event searches
        """        

        #==========================================================# 
        #Check if new atomic environment that have not been visited# 
        #==========================================================# 
        l_new_environement = self.new_environment()

        #===========================================================#
        #List of atom index on witch we will perform an event search#
        #===========================================================#
        l_atoms = []
        for id in l_new_environement :
            #list of atoms that have id in l_new_environment : 
            atom_idx =  [dict['atom index'] for dict in self.system.environment if dict['ID'] == id][0]
            #We select nsearch atoms randomly in this atom_idx 
            atom_idx = [random.choice(atom_idx) for _i in range(self.search_params['nsearch'])]
            #extend total list of atoms on which we gonna do an event search
            l_atoms.extend(atom_idx)

        #================================================#
        #For each atoms in atom_idx we do an event search#
        #================================================#
        l_fs = [self.pARTn_search(atom_index) for atom_index in l_atoms]

        #=================================================# 
        #For each results, we add the event to the catalog#
        #=================================================# 
        for fs in l_fs : 
            if fs is not None :
                dfevent_forward = fs[0]
                dfevent_backward = fs[1]
                energy_barrier = min(dfevent_forward['energy_barrier'], dfevent_backward['energy_barrier'])
                if self.search_params['emin_event'] < energy_barrier < self.search_params['emax_event'] : 
                    #Check if already in catalog : 
                    self.add_event_with_reconstruction(dfevent_forward, dfevent_backward)

    def search_without_reconstruction(self) : 
        """ 
        Event search procedure when `reconstruction == False`
        Made to be use with atomic environment style = 'cna' 
        For each non cristalline atom we launch nsearch event search
        """

        #============================================# 
        #List of atoms on which we do an event search# 
        #============================================# 

        #List of atoms that have non cristalline environement 
        l_atoms = [dict['atom index'] for dict in self.system.environment if dict['ID'] == 'noncrystal'][0]
        #for each atom in l_atoms we launch nsearch event searches 
        l_atoms *= self.search_params['nsearch']

        #==================================#
        #Launch len(l_atoms) event searches#
        #==================================#
        l_fs = [self.pARTn_search(atom_index) for atom_index in l_atoms]

        #===================================================# 
        #Loop over list results and add event to the catalog#
        #===================================================# 
        for i,fs in enumerate(l_fs) : 
            if fs is not None :
                dfevent = fs
                if self.search_params['emin_event'] < dfevent['energy_barrier'] < self.search_params['emax_event'] : 
                    #Check if event already in catalog : 
                    if len(self.system.catalog) > 0 : 
                        self.add_event_without_reconstruction(dfevent)
                    else : 
                        self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)


    def new_environment(self) : 
        """
        Find atomic environments id of the current step that have not been previously visited 

        Returns
        -------
        l_new_environment : list of str
            List of atomic environment ID of the current step that are new
        """
        #ID in the current system.environment 
        ids_current = [element['ID'] for element in self.system.environment]
        #remove 'crystal' environment
        try:
            ids_current.remove('crystal') #remove cystalline environment
        except ValueError:
            pass
        #Only select ID in ids_current that are not in visited_environment from previous step
        l_new_environments = [ids for ids in ids_current if ids not in list(self.system.visited_environment)]
        return l_new_environments 
    
    def compute_rate_Eyring(self, dE) : 
        """
        Compute the rate constant based on eq 11 of https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2019.00202/full 

        Parameters
        ----------
        dE : float
            activation energy

        Returns
        -------
        float
            rate constant
        """
        p = Parameters() 
        T = self.search_params['T'] 
        k0 = self.search_params['k0']
        return k0*((p.kb*T)/p.h)*m.exp(-dE/(p.kb*T))
    
    def add_event_with_reconstruction(self, dfevent_forward, dfevent_backward) : 
        """
        Search if dfevent_forward event is already in the catalog by checking topolgy ids 
        if not, add the event to the catalog, and the dfevent_backward event if it's not the same as the forward one 

        Parameters
        ----------
        dfevent_forward : pandas.Series
            Series with the forward event informations 
        dfevent_backward : pandas.Series
            Series with the backward event informations 
        """        

        #Only select rows with same event_id as dfenvent : 
        subset = self.system.catalog[self.system.catalog["event_id"] == dfevent_forward["event_id"]] 
        #subset of subset with rows with the same saddle_id : 
        subset = subset[subset["id_saddle"] == dfevent_forward["id_saddle"]]
        #subset of subset of subset with rows with the same final_id : 
        subset = subset[subset["id_final"] == dfevent_forward["id_final"]]
        #if there is no event with same IDs
        if len(subset) == 0 : 
            #add to the catalog foward reaction  
            self.system.catalog = pd.concat([self.system.catalog, dfevent_forward.to_frame().T], ignore_index=True)
            #Check if backward reaction is not the same as the forward one    
            if dfevent_forward["event_id"] != dfevent_forward["id_final"] :  
                self.system.catalog = pd.concat([self.system.catalog, dfevent_backward.to_frame().T], ignore_index=True)
            
    def add_event_without_reconstruction(self, dfevent) : 
        """
        Search if event is already in the catalog by checking, for a same atomic index, if the final positions are close.
        If not, add the event to the catalog.

        Parameters
        ----------
        dfevent : pandas.Series
            A pandas.Series with event informations
        """
        atol = 1e-3 
        rtol = 1e-3 

        #Only select rows with same atom index 
        subset = self.system.catalog[self.system.catalog["atom_index"] == dfevent['atom_index']]

        #Check if we have final positions of the event close to at least one final positions in the subset 
        if not subset["final_positions"].apply(lambda pos : np.allclose(pos, dfevent["final_positions"], atol=atol, rtol=rtol)).any() : 
            #if not add event to the catalog : 
            self.system.catalog = pd.concat([self.system.catalog, dfevent.to_frame().T], ignore_index=True)
            
    def event_series_with_reconstruction(self, min1_positions, saddle_positions, min2_positions, index_move, dE_forward, dE_backward) : 
        """
        Create forward and backward event pandas Series from an event search when `reconstruction == Truee`

        Parameters
        ----------
        min1_positions : (N,3) numpy.array of float
            positions of the first minimum
        saddle_positions : (N,3) numpy.array of float
            positions of at the saddle point
        min2_positions : (N,3) numpy.array of float
            positions of the second minimum
        index_move : int
            index of the atom that move the most
        dE_forward : float
            energy barrier of the forward reaction
        dE_backward : flat
            energy barrier of the backward reaction

        Returns
        -------
        dfevent_forward, dfevent_backward : (pandas.Series, pandas.Series)
            Series with : 
            - `'event_id'` : str 
                Topology id of the event 
            - `'initial_positions'` : (N,3) numpy.array of float 
                initial positions of the event 
            - `'saddle_positions'` : (N, 3) numpy.array of float 
                positions of the saddle point 
            - `'final_positions'` : (N,3) numpy.array of float 
                final positions of the event 
            - `'dE'` : float 
                activation energy of the event
            - `'k'`: float  
                rate constante 
            - `'id_saddle'`: str 
                Topology id of the saddle point 
            - `'id_final'`: str 
                Topology id of the final positions
            - '`move_atom_index'`
                index in `'initial_positions'`, `'saddle_positions'` and `'final_positions'`of the atom that move the most 
        """

        positions = [min1_positions, saddle_positions, min2_positions]
        cell = self.system.get_cell()
        #Create Needed event trajectory (for translation) 
        event_traj = [Atoms(positions=pos, cell=cell, pbc=True) for pos in positions]

        #Compute all needed topology ID : 
        id_min1 = pynauty.certificate(make_graph(event_traj[0], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])
        id_saddle = pynauty.certificate(make_graph(event_traj[1], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])
        id_min2 = pynauty.certificate(make_graph(event_traj[2], [index_move], self.atomenv_params['rnei'], self.atomenv_params['rcut'])[0])

        #Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr 
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2] 
        dx, dy, dz = ax/2 - min1_positions[index_move][0], ay/2 - min1_positions[index_move][1], az/2 - min1_positions[index_move][2]
        for atoms in event_traj : 
            atoms.translate(np.array([dx, dy, dz]))
            atoms.set_positions(atoms.get_positions(wrap=True))
        
        #neighbors lists around atom that move the most 
        ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
        rcutevent = self.search_params['rcutenv']

        dist = event_traj[0].get_distances(index_move, ind, mic=True)
        neighbor_list_forwward = np.where(dist<rcutevent)[0]
        dist = event_traj[2].get_distances(index_move, ind, mic=True)
        neighbor_list_backward = np.where(dist<rcutevent)[0]

        #Create event Series
        min1_positions = event_traj[0].get_positions() 
        saddle_positions = event_traj[1].get_positions() 
        min2_positions = event_traj[2].get_positions()

        dfevent_forward = pd.Series({'event_id' : id_min1 , 
                                     'initial_positions' : min1_positions[neighbor_list_forwward], 
                                     'saddle_positions': saddle_positions[neighbor_list_forwward], 
                                     'final_positions': min2_positions[neighbor_list_forwward], 
                                     'energy_barrier': dE_forward, 
                                     'k' : self.compute_rate_Eyring(dE_forward), 
                                     'id_saddle' : id_saddle, 
                                     'id_final': id_min2, 
                                     'move_atom_idx': np.where(neighbor_list_forwward == index_move)[0] })
        dfevent_backward = pd.Series({'event_id' : id_min2 , 
                                     'initial_positions' : min2_positions[neighbor_list_backward], 
                                     'saddle_positions': saddle_positions[neighbor_list_backward], 
                                     'final_positions': min1_positions[neighbor_list_backward], 
                                     'energy_barrier': dE_backward, 
                                     'k' : self.compute_rate_Eyring(dE_backward), 
                                     'id_saddle' : id_saddle, 
                                     'id_final': id_min1, 
                                     'move_atom_idx': np.where(neighbor_list_backward == index_move)[0] })
        
        return dfevent_forward, dfevent_backward
        


    def event_series_without_reconstruction(self, atom_index, final_positions, dE) : 
        """
        Create event pandas Series from an event search when `reconstruction == False`

        Parameters
        ----------
        atom_index : int
            atom index on which the event search have been made
        final_positions : (N,3) numpy.array of float 
            final positions of the event
        dE : float
            Energy barrier of the event

        Returns
        -------
        dfevent : pandas.Series 
            Series with : 
            - `'atom_index'` : atom index of the atom 
            - `'final_positions'` : final positions 
            - `'energy_barrier'` : energy barrier 
            - `'k'`: rate constant

        Notes
        ----- 
        The rate constant is calculated using the `compute_rate_Eyring(dE)` method 
        """         
        dfevent = pd.Series({'atom_index' : atom_index, 
                            'final_positions' : final_positions, 
                            'energy_barrier' : dE,
                            'k' : self.compute_rate_Eyring(dE)})
        return dfevent

    def pARTn_search(self, atom_index) : 
        """
        Use pARTn with Lammps to find new event

        Parameters
        ----------
        atom_index : int
            index of the central atom on which we perform the event search

        Returns
        -------
        pandas.Series or tuple or None 
            if pARTn error or if `delr1` or `delr2` are both greater than 0.2, return `None`
            if `reconstruction == False` return : 
                dfevent : pandas.Series 
                    Series with : 
                    - `'atom_index'` : atom index of the atom 
                    - `'final_positions'` : final positions 
                    - `'energy_barrier'` : energy barrier 
                    - `'k'`: rate constant
            else return : 
                dfevent_forward, dfevent_backward : (pandas.Series, pandas.Series)
                    Series with : 
                    - `'event_id'` : str 
                        Topology id of the event 
                    - `'initial_positions'` : (N,3) numpy.array of float 
                        initial positions of the event 
                    - `'saddle_positions'` : (N, 3) numpy.array of float 
                        positions of the saddle point 
                    - `'final_positions'` : (N,3) numpy.array of float 
                        final positions of the event 
                    - `'dE'` : float 
                        activation energy of the event
                    - `'k'`: float  
                        rate constante 
                    - `'id_saddle'`: str 
                        Topology id of the saddle point 
                    - `'id_final'`: str 
                        Topology id of the final positions
                    - '`move_atom_index'`
                        index in `'initial_positions'`, `'saddle_positions'` and `'final_positions'`of the atom that move the most 
        """
        #Logs
        logging.basicConfig(filename='pykmc.log', filemode='a', level=logging.DEBUG, format='%(message)s')

        #Setup Lammps :
        lmp = lammps(cmdargs=['-screen', 'none'])
        artn = pypARTn2.artn(engine='lmp')
        initialize_default_lammps(self.system, lmp)

            #Potential : 
        lmp.command('pair_style {}'.format(self.potential['pair_style']))
        lmp.command('pair_coeff {}'.format(self.potential['pair_coeff']))
        
        lmp.command("plugin load {}".format(self.search_params['path_artnso']))
        lmp.command("fix 10 all artn dmax {}".format(self.search_params['partn_dmax']))
        lmp.command("min_style fire")

        #SETUP ARTN
        artn.set('engine_units', 'lammps/metal')
        artn.set('verbose',self.search_params['partn_verbose'])
        artn.set('struc_format_out', 'none')
        artn.set("lpush_final", True)
        artn.set("lmove_nextmin", False) #if true fortran runtime error when event not found
        artn.set("ninit", self.search_params['partn_ninit'])
        artn.set("forc_thr", self.search_params['partn_forc_thr'])
        artn.set('push_mode', self.search_params['partn_push_mode'])
        if self.search_params['partn_push_mode'] == 'rad' : 
            artn.set('push_dist_thr', self.search_params['partn_push_dist_thr'])
        artn.set("push_step_size",  self.search_params['partn_push_step_size'])
        artn.set("push_ids", [atom_index])
        artn.set('eigen_step_size', self.search_params['partn_eigen_step_size'])
        artn.set('lanczos_disp', self.search_params['partn_lanczos_disp'])
        artn.set('nsmooth',  self.search_params['partn_nsmooth'])
        artn.set('nperp', self.search_params['partn_nperp'])

        #Run
        lmp.command("minimize 0 1e-3 1000 1000")

        #Need to extract min 1, min 2, saddle positions and energy barrier
        err = artn.get_runparam("error_message")
        if not err :
            #Results 
            delr1 = artn.extract('delr_min1') 
            delr2 = artn.extract('delr_min2')

            #Checks if one minimum is close to the original configuration 
            if delr1 < self.search_params['partn_delr_threshold'] or delr2 < self.search_params['partn_delr_threshold'] : 

                E_sad = artn.extract("etot_sad")
                E_min1 = artn.extract("etot_min1")
                E_min2 = artn.extract("etot_min2")

                dE_forward = E_sad - E_min1 
                dE_backward = E_sad - E_min2 
    
                min1positions = artn.extract("tau_min1")
                min2positions = artn.extract("tau_min2")
                saddlepositions = artn.extract("tau_sad")

                #Generate event pandas Series
                rcutevent = self.search_params['rcutenv']
                ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
                if self.reconstruction :
                    #Find atoms that move the most 
                    dist = (min1positions-saddlepositions)**2
                    dist = dist.sum(axis=-1)
                    dist = np.sqrt(dist)
                    dist[dist > rcutevent] = 0 #if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
                    index_move = np.argmax(dist)

                    dfevent_forward, dfevent_backward = self.event_series_with_reconstruction(min1positions, saddlepositions, min2positions, index_move, dE_forward, dE_backward)

                    return dfevent_forward, dfevent_backward
                else :  
                    #Find neighbors of atom_index
                    dist = self.system.get_distances(atom_index, ind, mic=True)
                    neighbor_list = np.where(dist<rcutevent)[0]

                    #Create event Series 
                    if delr1 < delr2 : #meaning min1 close to initial coordinates : 
                        final_positions = min2positions[neighbor_list]
                        dE = dE_forward
                    else : #min2 close to initial coordinates
                        final_positions = min1positions[neighbor_list]
                        dE = dE_backward 

                    return self.event_series_without_reconstruction(atom_index, final_positions, dE)
            else :
                return None
        else : 
            return None
        


#    def dimer_search(self, atom_index, potential): 
#        """ 
#        Use Dimer search with ASE and lammps
#        """
#        # Set up LAMMPS calculator
#        run('export ASE_LAMMPSRUN_COMMAND=/Users/hugomoison/Programmes/lammps-29Aug2024/src', shell=True)
#        lammps_command = ["lmp_mpi"]  
#        lammps_parameters = {'pair_style': 'eam', 'pair_coeff': ['* * ./Ni_v6_2.0_LKBeland2016.eam Ni']}
#        files = ['Ni_v6_2.0_LKBeland2016.eam']
#        #initial potential energy : 
#        lammps_calc = LAMMPS(files=files , lammps_command=lammps_command,**lammps_parameters)
#
#        atoms = Atoms(positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
#        atoms.calc = lammps_calc
#
#        # Calculate the energy of the system with LAMMPS
#        atoms.get_potential_energy()
#        #setup dimer : 
#        dcontrol = DimerControl(logfile='dimer_search.log', initial_eigenmode_method='displacement', displacement_method='vector', displacement_center=atom_index, displacement_radius=4.0)
#        d_atoms = MinModeAtoms(atoms, dcontrol)










