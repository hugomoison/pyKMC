#!/bin/bash


################################ Main Input/Output files ##########################################

export INI_FILE_NAME='kARTdata'            # The file name containing the intial configuration 
export CONFFILE='allconf.xyz'              # Name of the file to store all the configurations visited (default: allconf)

###################################### Simulation Details #########################################

export NBRE_KMC_STEPS=10                   # The max number of KMC steps to be executed
export TOTAL_TIME=3600.0                   # The maximum simulation time in seconds 
export TEMPERATURE=300.0                   # The simulated temperature in kelvin

export NUMBER_ATOMS=1022                   # The total number of atoms 
export SIMULATION_BOX=28.5530              # The size of the simulation box (x, y and z)
export NSPECIES=2                          # The number of different atom types (default: 2)
export ATOMIC_SYMBOLS="Fe:inactive C"      # The list of elements

###################################### Restart options ############################################

export RESTART_KMC=".false."               # IF true, restart from previous run
export RESTART_FILE="this_conf"            # The file name used to continue a simulation from where it was last stopped
export RESTART_IMPORT=".false."            # Start a NEW simulation but with the current KMC event catalogue (events.uft and topos.list)
export NEW_CATALOGUE=".false."             # IF true, will continue simulation but will rebuild event catalogue from scratch

#################################### Basin parameters #############################################

export MIN_SIG_BARRIER=0.8                 # Max height of barrier and inv. barrier for an event to be considered inside a basin

#################################### Topology Params ##############################################

export TOPO_RADIUS=6.0                     # radius for topology cluster 
export MAX_TOPO_CUTOFF=2.7                 # length-cutoff used by default to link two atoms  
export MIN_TOPO_CUTOFF=2.0 
#export CRYST_TOPOID=746699                # topo id of the crystalline-like topologies
export CRYST_TOPO_RADIUS=4.0               # radius for crystal-like topologies (default: 4.0 A)

#################################### Force calculations ###########################################

export ENERGY_CALC="LAM"                   # choose between EDP or SWP or SWA or LAM (Lammps)
export INPUT_LAMMPS_FILE='in.lammps'       # Name of the input file for lammps (default name : in.lammps )




################# ART PARAMETERS ##################################################################

export RADIUS_INITIAL_DEFORMATION=3.0      # Cutoff for local-move (in angstroems)
export EXIT_FORCE_THRESHOLD=0.01           # Threshold for convergence at saddle point
export FINE_EXIT_FORCE_THRESHOLD=0.01      # finner Threshold for convergence at saddle point 

export EIGENVALUE_THRESHOLD=-1.0           # Eigenvalue threshold for leaving basin
export MAX_PERP_MOVES_BASIN=2              # Maximum number of perpendicular steps leaving basin
export MIN_NUMBER_KSTEPS=2                 # Min. number of ksteps before calling lanczos
export FORCE_THRESHOLD_PERP_REL=0.01       # Threshold for perpendicular relaxation

export INITIAL_STEP_SIZE=1.00              # Size of initial displacement, in A
export BASIN_FACTOR=3.00
export MIN_NUMBER_KSTEPS=3                 # Min. number of ksteps before calling lanczos
export MAX_PERP_MOVES_BASIN=6              # Maximum number of perpendicular steps leaving basin
export MAX_PERP_MOVES_ACTIV=20             # Maximum number of perpendicular steps during activation
export MAX_ITER_BASIN=40                   # Maximum number of iteraction for leaving the basin (kter)
export DTMAX_FIRE=0.25                     # default value

export NUMBER_LANCZOS_VECTORS=15           # Number of vectors included in lanczos procedure
export LANCZOS_STEP=1e-7                   # Size of the step for the numerical derivative (def: 0.001)
#export CHECK_LANCZOS_STAB=".true."        # check lanczos stability over 200 steps, each iteration uses previous lanczos 


#################################### GENERIC events parameters ########################
export SEARCH_FREQUENCY=10                 # Minimum number of attempts to find a GENERIC event per new topology encountered
export THRES_INCREASE_FREQ=25              # Number of failed attempts encountered because increasing the EIGEN_THRESH
export TYPE_EVENT_UPDATE="SPEC"            # choose between SPEC or GENE 
export USE_LOG_SEARCH=".false."            # Search frequency is multiplied by logarithmic increasing function (default .true.)
export CHECK_INI_SAD_CONNECTIVITY=".true."


############### Printing details ######################################################################
export ALLCONF_WITH_SADDLE=".true."
export PRINT_DETAILS=".true."              # Prints the details of activation and minimization 
export MINSAD_DETAILS=".false."            # Prints the details of activation and minimization 
export USE_TXT_EVENTFILE=".true."
export STATISTICS=".true."                 # Write statistics about force and event calculation  
export OUTPUT_CONFIG_EVENTS=".true."       # IF true, will create a txt file with the list of all the topologies and events after each KMC step
export OUTPUT_SPECIFIC=".true."
#export OUTPUT_NEB_GEN_EVENT=".true."      # Can be useful






