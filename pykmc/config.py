from dataclasses import dataclass
import configparser

#TODO add method to generate documentation parameters
#TODO add check if parameters don't clash

@dataclass 
class Parameters : 
   """
   Physical parameters in lammps metal units
   """ 
   #Boltzmann constant
   kb = 8.6173303e-05 #eV.K^-1
   #Planck constant 
   h = 4.135667696e-03 #eV.ps 



DEFAULT = {
    'Control' : {
        'config_file' : None, 
        'output_file' : 'trajkmc.xyz',
        'catalog' : None,
        'dimension' : 3,
        'nprocs' : 1, 
        'backend' : 'local',
        'reconstruction' : True

    }, 
    'Potential' : { 
        'file' : None
    },
    'Minimization' : {
        'style' : 'lammps', 
        'min_style': 'cg', 
        'etol': 1.0e-6, 
        'ftol': 1.0e-8, 
        'maxiter' : 100, 
        'maxeval' : 1000
    }, 
    'AtomicEnvironment' : {
        'radd_cna' : 0
    },
    'EventSearch' : {
        'emin_event' : 0.2,
        'emax_event' : 6, 
        'partn_dmax' : 6.0, 
        'partn_verbose' : 2, 
        'partn_ninit' : 2, 
        'partn_forc_thr' : 0.01,
        'partn_push_mode' : 'rad', 
        'partn_push_dist_thr' : 3.0, 
        'partn_push_step_size' : 0.4, 
        'partn_eigen_step_size' : 0.2, 
        'partn_lanczos_disp' : 0.0005,
        'partn_nsmooth' : 3, 
        'partn_nperp' : 5,
        'partn_delr_threshold' : 0.5,
        'k0' : 1.0e-12
    }, 
    'PSR' : {
        'kmax_factor' : 1.8,
        'hausdorff_dist_thr' : 0.1,
    }
}


DESCRIPTIONS = {
    "Control" : {"__description__" : ("The following parameters are general parameters that control the KMC simulations and resources used."),
                 "nkmc_steps" : "number of KMC steps",
                 "config_file" : "Path to the initial configuration file",
                 "output_file" : "Path to the file where the trajectory is written, format must be recognized by ase.io.write()", 
                 "catalog" : "Path to a catalog to reuse from a previous simulation",
                 "dimension" : "Dimension of the system",
                 "nprocs" : "number of MPI process to use",
                 "backend" : "if running the simulation locally (`'local'`), or on a cluster (`'slurm_allocation'`)", 
                 "reconstruction" : "if a new catalog is generated at each step or reused"
    },
    "Potential" : {"__description__" : ("The potential section deals with the description of the potential in the format of the E/F engine used, for lammps :"), 
                   "file" : "path to the potential file if needed", 
                   "pair_style" : "lammps pair style command", 
                   "pair_coeff" : "lammps pair coeff command"
                   }, 
    "Minimization" : {"__description__" : ("Parameters used for the minimization of the system"), 
                      "style" : "E/F engine used", 
                      "min_style" : "lammps minimization style", 
                      "etol" : "lammps stopping tolerance for energy",
                      "ftol" : "lammps stopping tolerance for force", 
                      "maxiter" : "lammps max iterations of minimizer", 
                      "maxeval" : "lammps max number of force/energy evaluations"
                      },
    "AtomicEnvironment" : {"__description__" : ("Parameters used to attribute an atomic environment to each atoms"), 
                           "style" : "style used, `cna`, `graph` or `cna/graph`", 
                           "rnei" : "maximal distance to consider that two atoms are connected when `graph` and `cna/graph` is used",
                           "rcut" : "radial cuttoff defining the environment around an atom, used when `graph` and `cna/graph`",
                           "radd_cna" : "when `cna/graph` is used, graph for atoms at a distance inferior to radd_cna to a atom having a non crystalline environment is also computed"

    }, 
    "EventSearch" : {"__description__" : ("Parameters related to the search of transition paths"),
                     "style" : "which method used, `'pARTn'` for ARTn method", 
                     "rcutenv" : "radius of the sphere defining the positions of atoms, around the central atom, that are saved in the catalog",
                     "nsearch" : "number of event search for each different atomic environments",
                     "path_artnso" : "path to the partn library file",
                     "emax_event" : "event found having a higher barrier energy value are not saved to the catalog",
                     "emin_event" : "event found having a lower barrier energy value are not saved to the catalog",
                     "k0" : "value of k0 when computing the rate constant with $k0 \frac{k_{b}T}{h}e^{-\frac{dE}{k_{b}T}}",
                     "T" : "Temperature of the simulation"
                     },
    "PSR" : {"__description__" : ("Parameters controlling point set registration (shape matching)"), 
             "style" : "which method is used, `'ira'` for IterativeRotationsAssignments", 
             "kmax_factor" : "factor for multiplication of search radius", 
             "hausdorff_dist_thr" : "Hausdorff distance threshold representing the largest displacement of any atom from the reference structure after IRA point set registration"
             }
}


@dataclass 
class SystemConfig : 
    """ 
    Class to manage input parameters
    """ 
    @staticmethod 
    def from_file(config_file: str): 
        """
        Read input parameters from file

        Parameters
        ----------
        config_file : str
            path to the input file
        Returns
        -------
        config_dict : dict 
            dictionary of input parameters

        Raises
        ------
        Exception
            if wrong section in input file
        """
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation()) 

        config.optionxform = str #Enable uppercase variable
        try : 
            with open(config_file) as f : #So we can raise an exception
                config.read(config_file)
        except :
            raise Exception('No configuration file found')
        
        #Check 
        if not config.has_section('Control')  : 
            raise Exception('Control section in configuration file is mandatory')
        if not config.has_section('Potential') : 
            raise Exception('Potential section in configuration file is mandatory')
        if not config.has_section('Minimization') : 
            raise Exception('Minimization section in configuration file is mandatory')
        if not config.has_section('EventSearch') : 
            raise Exception('EventSearch section in configuration file is mandatory')
        if not config.has_section('PSR') : 
            raise Exception('PSR section in configuration file is mandatory')


        
        #Convert ConfigParser in a dictionaries of dictionary and convert values 
            #Initialize with default values 
        config_dict = {section: default for section, default in DEFAULT.items()}
            #Add config values
        for section in config.sections() : 
            if section not in config_dict: 
                config_dict[section] = {}
            config_dict[section].update({
                    key : SystemConfig._convert_value(config.get(section,key))
                    for key, _ in config.items(section)
            })

        return config_dict
    
    def _convert_value(value) : 
        """ 
        Convert str value from the input file to int, float, boolean or str

        Parameters
        ---------- 
        value : 
            value to be converted 

        Returns
        ------- 
        value : 
            the value after the conversion
        """
        try :
            if value == 'True' : 
                return True 
            if value == 'False' :  
                return False
        except ValueError : 
            pass
        try : 
            if '.' not in value and 'e' not in value and 'E' not in value : 
                return int(value) 
        except ValueError : 
            pass
        try : 
            return float(value)
        except ValueError : 
            pass 
         
        return value

