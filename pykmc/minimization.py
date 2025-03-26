from lammps import lammps 
from ase.io.lammpsdata import write_lammps_data
import numpy as np 
from subprocess import run
from executorlib import SingleNodeExecutor
from .utilities import initialize_default_lammps

class Minimization:
    """
    Define and run the minimization procedure

    Attributes
    ----------
    system : System Object 
        the System on which we perfom the minimization 
    minimization_style : str 
        the minimization style use, can be 'lammps' 
    minimization_params : dict
        all commands needed by the program used in minimization_style to execute the minimization
    potential : dict
        commands to define the potential used by the program defined by minimization_style
    dimension : int, optional
        dimension of the system, by default 3
    nprocs : int, optional
        number of procs available, by default 1
    backend : str, optional
        parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

    Methods 
    ------- 
    run()
        run the minimization and update the System positions
    minimize_lammps() 
        run lammps to perform the minimization
    """     

    def __init__(self, system, minimization_style, minimization_params, potential, dimension, nprocs, backend) : 
        self.system = system
        self.minimization_style = minimization_style
        self.minimization_params = minimization_params
        self.potential = potential
        self.dimension = dimension
        self.nprocs = nprocs
        self.backend = backend

    def run(self) : 
        """
        Execute minimization based on minimization_style
        """

        #============================================#
        #Run minimization based on minimization_style#
        #============================================#
        with SingleNodeExecutor() as exe :
            match self.minimization_style : 
                case "lammps":
                    fs = exe.submit(self.minimize_lammps, resource_dict={"cores": self.nprocs})
                case _:
                    self.system.logger.logger.error('ERROR:Minimization style not known')
                    raise Exception("Minimization style not known")

        #======================================#
        #Get result and update system.positions#
        #======================================#
        if self.nprocs == 1 : 
            positions = fs.result()
        else : 
            positions = fs.result()[0]
        positions[positions < 0] = 0 #This is because I can have small negative number and it messes up with kdtree
        self.system.set_positions(positions)

    def minimize_lammps(self) : 
        """
        Run a Lammps minimizatin based on the minimization_params and potential

        Returns
        -------
        positions : (N,3) np.array of float
            positions after the minimization if `rank == 0` else `None`
        """        

        from mpi4py import MPI
        #MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Initialize lammps :
        lmp = lammps(cmdargs=['-screen', 'none'])
        initialize_default_lammps(self.system, lmp)

        #Potential
        lmp.command('pair_style {}'.format(self.potential['pair_style']))
        lmp.command('pair_coeff {}'.format(self.potential['pair_coeff']))
        #Minimization 
        lmp.command('min_style {}'.format(self.minimization_params['min_style']))
        lmp.command('minimize {} {} {} {}'.format(self.minimization_params['etol'], self.minimization_params['ftol'], self.minimization_params['maxiter'], self.minimization_params['maxeval']))
        #gather all positions 
        positions = lmp.gather_atoms("x", 1, 3)

        if rank == 0 : 
            #convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            #clean files
            run('mv log.lammps log.minimize_lammps', shell=True)
            return positions 
        else : 
            return None
