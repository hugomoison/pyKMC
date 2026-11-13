# User Guide 

_This is a temporary user guide that aim to provide basic information on the simulation workflow and how to choose simulation's parameters_ 

pyKMC is an on-the-fly kinetic Monte Carlo (KMC) program.

At each step, if a new atomic environment is encountered, pyKMC performs event searches and adds the resulting generic events to a reference event table.
When a previously visited environment is found again, pyKMC refines the stored reference events to account for elastic deformations, and builds an active event table specific to the current state.

An event is then selected from this active table based on the chosen KMC algorithm and applied to the system, advancing the simulation.

To start a simulation, pyKMC requires a configuration file in the INI format. This file contains general simulation parameters, as well as tool-specific settings needed to run the different stages of the simulation workflow. 

## Control 

The first section of the INI configuration file, called [Control], defines general simulation parameters.
You must provide an initial atomic configuration file, readable by ase.io.read, which includes atomic positions, cell parameters, and periodic boundary conditions (PBC). For example, if using an XYZ file, it should look like:

```bash 
2047
Lattice="28.16 0.0 0.0 0.0 28.16 0.0 0.0 0.0 28.16" pbc="T T T"
Ni       0.00000000       0.00000000       0.00000000       
Ni       0.00000000       1.76000000       1.76000000       
... 
``` 
The path to this file should be provided using the `initial_config` key. 
You must also specify : 
- the number of KMC steps to run with `n_steps`
- the simulation engine to use to compute energy, forces and perform event searches/refinements (e.g., lammps) with the `engine` key

A minimal example of a `[Control]` section would be:
```INI 
[Control] 
initial_config = myconfig.xyz 
n_steps = 100 
engine = lammps
``` 
Additional options in this section allow you to customize output filenames (see the full parameter documentation).
pyKMC also saves the reference event table and the list of visited environments as .pickle files. To reuse them in a new simulation, simply provide their paths:

```INI 
reference_table = my_reference_table.pickle 
visited_environments = my_visited_environments_list.pickle
```  
Alternatively, you may provide only a list of visited environments if you wish to exclude certain environments from being explored.

## Engine 

The INI configuration file must also include a section specific to the engine you selected in the `[Control]` section.
For example, if you set `engine = lammps`, your file should include a `[Lammps]` section.

### Lammps 

When using LAMMPS, you need to specify the potential parameters.
Currently, only pair potentials are supported. You must provide the `pair_style` and `pair_coeff` keys

Additionally, you can change default parameters used during system minimization (see the full parameter documentation). 

A minimal `[Lammps]` section might look like:
```INI 
[Lammps]
pair_style = eam/alloy 
pair_coeff = * * my_file.eam Ni 
``` 

`pair_coeff` can also be a list in square brackets, with elements separated by commas: pyKMC sends one
`pair_coeff` command per element, in order. Hybrid styles need this, since LAMMPS loads them with several
`pair_coeff` lines, each naming its sub-style. `pair_style` is written as in a LAMMPS input.
```INI
[Lammps]
pair_style = hybrid/overlay eam/alloy zbl 1.0 2.0
pair_coeff = [* * eam/alloy my_file.eam Ni, 1 1 zbl 28.0 28.0]
```
sends
```
pair_style hybrid/overlay eam/alloy zbl 1.0 2.0
pair_coeff * * eam/alloy my_file.eam Ni
pair_coeff 1 1 zbl 28.0 28.0
```
A value without brackets is one `pair_coeff` command, sent as written.

## Atomic Environment 

During the simulation, at each step, pyKMC assigns an atomic environment ID to every atom in the system.
This ID serves as a unique fingerprint of the atom’s local environment, allowing pyKMC to identify recurring configurations.

Each reference event is also tagged with an environment ID, corresponding to the initial local configuration of the event’s central atom—the atom that moves the most during the transition. For events, IDs are always computed using the graph style (see below).

To determine whether a stored reference event can be reused, pyKMC compares the event's environment ID with those computed in the current system.

Different ID generation strategies (called styles) are available to define atomic environments. 

To define parameters related to the generation of those IDs, the INI configuration file should contain an `[AtomicEnvironment]` section. 

The two main parameters are specified by the `rnei` and `rcut` keys. `rnei` defines the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors. `rcut`defines the atomic environment sphere, atoms in this sphere are part of the local atomic environment. 

You can select the method (or style) used to assign an atomic environment ID to each atom.
The available styles are:
- cna : 
In this mode, pyKMC counts the number of neighbors around each atom and checks whether it matches a typical crystalline coordination number (6, 8, or 12).
    - if the atom has one of these neighbor counts, it is labeled "crystal".
    - Otherwise, it is labeled "noncrystal".
<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="images/atomic_env_base.png" width="220" />
  <div style="text-align: center; font-weight: bold;">
    Using style=cna gives: 
  </div>
  <img src="images/atomic_env_cna.png" width="300" />
</div>

- graph : 
In this style, pyKMC constructs a graph from each atom's environment using pyNauty. 
A unique, canonical certificate (a binary) is then computed using pyNauty, serving as the atom’s environment ID.
<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="images/atomic_env_base.png" width="220" />
  <div style="text-align: center; font-weight: bold;">
    Using style=graph gives: 
  </div>
  <img src="images/atomic_env_graph.png" width="300" />
</div>

- cna/graph : 
In large systems, many atoms are in a perfect crystalline environment. Computing graph-based IDs for all of them is inefficient.
This hybrid mode provides a compromise: 
    - First, a CNA classification is performed. 
    - If an atom is labeled "crystal", it keeps this simple ID. 
    - If it is labeled "noncrystal", a graph-based ID is computed. 
<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="images/atomic_env_base.png" width="220" />
  <div style="text-align: center; font-weight: bold;">
    Using style=cna/graph gives: 
  </div>
  <img src="images/atomic_env_cnagraph.png" width="300" />
</div>

When searching for an event around a "noncrystal" atom, it may happen that another atom ends up being the one that moves the most. In this case, the resulting event will be tagged with the graph ID of that other atom.
If this ID corresponds to a "crystal" atom in the current system, that event will never be selected. 

To avoid this issue, you can instruct pyKMC to expand the graph style to the nth neighbors of the "noncrystal" atom using the `neighbors_add` key. 

For example, when looking at a vacancy with `neighbors_add = 1` it gives: 

<div style="text-align: center;">
<img src="images/atomic_env_radd.png" width="400" />
</div>

Finally, the `[AtomicEnvironment]` section of the INI configuration file will look like this : 
```INI 
[AtomicEnvironment] 
style = cna/graph
rnei = 3.0 
rcut = 6.5 
neighbors_add = 1 
``` 
## Event Search 

The `[EventSearch]` section lets you define:
- the algorithm used to search for new events,
- the criteria used to keep or discard events in the reference event table,
- if a refinement is successful.

This section requires two mandatory keys:
- `style` : the algorithm used to perform event searches. _Currently, only partn is supported._
- `nsearch` : number of event searches to perform per atomic environment.

A minimal `[EventSearch]` section looks like:
```INI 
[EventSearch] 
style = partn 
nsearch = 50 
``` 

When an event is found, it is characterized by two energy barriers, a forward energy barrier $dE_{forward}$ and an inverse energy barrier $dE_{backward}$ : 
<div style="text-align: center;">
  <img src="images/pesevent.png" width="400" />
  <div style="font-size: 0.9em; color: gray; margin-top: 5px;">
    Potential energy surface
  </div>
</div>

The event is added to the reference table only if it satisfies all the following conditions:

- $dE_{forward}$ < `emax_event` 
- $dE_{forward}$ > `emin_event` 
- $dE_{backward}$ > `emin_event` 
- it is not highly asymmetric: the event is rejected if both $dE_{forward}$ > `energy_asymmetry` x `backward_emin_event` and $dE_{backward}$ < `backward_emin_event`. 

Once a reference event is reused, it is refined to adapt to the current atomic configuration. The refinement is considered successful if:
- The central atom moves less than `refined_minimum_delr_thr` between the current position and the refined minimum.
- The difference in energy barriers between the generic and refined events is less than `refined_energy_thr`.
These thresholds ensure that the refined event remains consistent with the original.

To further control the behavior of the selected event search algorithm (style), you must define a separate section matching the algorithm name. For instance, if you choose `style = partn` you must also include a `[pARTn]` section in your INI file to configure specific parameters for the pARTn method. 

### pARTn 

All other parameters have default values (see the full parameter documentation), but depending on your system, you may need to adjust some of them for optimal performance.
Parameters related to refinements are prefixed with r_

A minimal `[pARTn]` section will look like this : 
```INI 
[pARTn] 
delr_thr = 0.1
r_eigval_thr = -0.02
```

For a detailed explanation of the pARTn algorithm and its parameters, please refer to the official pARTn documentation.

## Rate Constant 

Each time an event is added to either the reference or the active event table, a rate constant is computed. The method used to compute this rate is defined in the [RateConstant] section of the INI configuration file.

Currently, the only implemented style is : `style = constant`. This method computes the rate constant using the following equation. 

$$
k = k_{0} e^{-\frac{dE_{forward}}{{k_{b}T}}}
$$
Where:
- $k$ is the rate constant,
- $k_{0}$ is a user-defined pre-exponential factor (typically $10^{13}s^{-1}=10ps^{-1}$),
- $T$ is the system temperature, defined by the user.
- $dE_{forward}$ is the forward energy barrier,
- $k_{b}$​ is the Boltzmann constant.

All parameters must be provided in LAMMPS metal units ($k_{0}$ in $ps^{-1}$, $T$ in K).

Typically, it gives:

```INI 
[RateConstant]
style = constant 
k0 = 10
T = 300 
``` 

## PSR 

When applying a reference event to a system for refinement, pyKMC performs a point set registration (also known as shape matching) to align the atomic positions from the reference event with the local atomic environment of the atom currently targeted for refinement.

You can define the algorithm to use with the `style` key. _Currently only 'ira' is implemented_.  This is the only mandatory parameter. 

You may also want to adjust the default value of matching_score_thr, which sets the maximum allowed matching score for a successful registration.
For `style = ira`, the matching score corresponds to the Hausdorff distance between the two point sets.

This gives:
```INI 
[PSR]
style = ira 
matching_score_thr = 0.3 
``` 

As with the Engine and EventSearch parts, you must also include a dedicated section for the selected point set registration method.

### IRA 

When using IRA, you can change the default parameters. For a full explanation, please refer to the full parameters documentation and the IRA documentation. 
To ensure access to default values, the section must be present in the configuration file, even if it's empty.

## Final configuration file 

Finally, based on the previous examples, a complete INI configuration file would look like this:

```INI 
[Control] 
initial_config = myconfig.xyz 
n_steps = 100 
engine = lammps

[Lammps]
pair_style = eam/alloy 
pair_coeff = * * my_file.eam Ni 

[AtomicEnvironment] 
style = cna/graph
rnei = 3.0 
rcut = 6.5 
neighbors_add = 1  

[EventSearch] 
style = partn 
nsearch = 50  

[pARTn] 
delr_thr = 0.1
r_eigval_thr = -0.02

[RateConstant]
style = constant 
k0 = 10 
T = 300  

[PSR]
style = ira 
matching_score_thr = 0.3  

[IRA] 
```

## How to restart a simulation

When a simulation completes its `n_steps` KMC steps, pyKMC writes a restart file named `restart_<last_step>.npz` (e.g. `restart_100.npz`). This file stores the last KMC step, the simulated time reached and the elapsed wall time.

During the simulation, pyKMC also writes:

- the trajectory (`trajkmc.xyz` by default, see `trajectory_output`),
- the reference event table (`reference_table.pickle` by default, see `reference_table_output`),
- the list of visited environments (`visited_environments.pickle` by default, see `visited_environments_output`).

To restart, give the restart file with the `restart_file` key in the `[Control]` section. You can set `initial_config` directly to the trajectory file: pyKMC always reads the last configuration of the file. To reuse the events and environments that were already explored, also give the reference table and the visited environments:

```INI 
[Control] 
initial_config = trajkmc.xyz 
restart_file = restart_100.npz 
reference_table = reference_table.pickle 
visited_environments = visited_environments.pickle 
n_steps = 100 
engine = lammps
``` 

On restart:

- `n_steps` is the number of **additional** steps to run. In the example above, the simulation continues from step 101 to step 200.
- The step counter, the simulated time and the wall time continue from the values in the restart file.
- The system is not minimized again: the positions read from `initial_config` are kept as they are.
- The trajectory and output files are appended to instead of being overwritten.

_A restart file is written only when all `n_steps` steps are completed. It is not written if the simulation stops earlier._

## Running a simulation

pyKMC requires MPI and must be run with at least two MPI ranks: rank 0 runs the main KMC loop, and the other ranks run the LAMMPS instance(s).

Once both the input file and the initial configuration file are ready, launch the simulation by executing:

```bash 
mpirun -n 2 python -m pykmc -in <your_input_file_name> 
``` 

## Using multiple lammps instances 

To speed up event searches and the refinement part of the KMC loop, it is possible to use multiple LAMMPS instances.
Each instance runs independently on a separate set of cores.

The number of instances is defined by the `n_sessions` parameter in the `[Control]` section of the input file.
The main KMC loop always runs on rank 0, and the LAMMPS instances share the remaining ranks. You therefore need at least `n_sessions + 1` MPI ranks.

For example, if you have 8 cores available and want to run 4 LAMMPS instances, your input file should contain the following parameters:

```INI 
[Control]
...
n_sessions = 4 
... 
``` 

You can then start the simulation with:
```bash 
mpirun -n 8 python -m pykmc -in <your_input_file> 
``` 
The available cores will be automatically split according to your configuration.
In this example, the LAMMPS instances will run on ranks 1–2, 3–4, 5–6, and 7, while the main KMC loop runs on rank 0.















