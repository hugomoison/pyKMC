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
- the simulation engine to use to compute energy, forces and perfrom event searches/refinements (e.g., lammps) with the `engine` key

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

## Atomic Environment

During the simulation, at each step, pyKMC assigns an atomic environment ID to every atom in the system.
This ID serves as a unique fingerprint of the atom’s local environment, allowing pyKMC to identify recurring configurations.

Each reference event is also tagged with an environment ID, corresponding to the initial local configuration of the event’s central atom—the atom that moves the most during the transition. For the event, their ID are always computed based on graph (see below).

To determine whether a stored reference event can be reused, pyKMC compares the event's environment ID with those computed in the current system.

Different ID generation strategies (called styles) are available to define atomic environments.

To define parameters related to the generation of those IDs, the INI configuration file should contains a `[AtomicEnvironment]` section.

The two main parameters are specified by the `rnei` and `rcut` keys. `rnei` defines the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors. `rcut`defines the atomic environment sphere, atoms in this sphere are part of the local atomic environment.

You can select the method (or style) used to assign an atomic environment ID to each atom.
The available styles are:
- cna :
In this mode, pyKMC counts the number of neighbors around each atom and checks whether it matches a typical crystalline coordination number (6, 8, or 12).
    - if the atom has one of these neighbor counts, it is labeled "crystal".
    - Otherwise, it is labeled "noncrystal".
This style should be only use when setting in the `[Control]` section `reconstruction = False`. (_Not working anymore for the moment_).
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

For example, when looking at a vacancy with `neighbors_add = 1` it will gives :

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
## Event Search :

The `[EventSearch]` section lets you define:
- the algorithm used to search for new events,
- the criteria used to keep or discard events in the reference event table,
- if a refinement is successful.

This section requires two mandatory keys:
- `style` : the algorithm used to perform event searches. _Currently, only partn is supported._
- `nsearch` : number of event searches to perform per atomic environment.

A minimal configuration looks like:
A `[EventSearch]` section in the INI configuration file is typically :
```INI
[EventSearch]
style = partn
nsearch = 50
```

When an event is found, it is characterized by two energy barriers, a foward energy barrier $dE_{foward}$ and an inverse energy barrier $dE_{backward}$ :
<div style="text-align: center;">
  <img src="images/pesevent.png" width="400" />
  <div style="font-size: 0.9em; color: gray; margin-top: 5px;">
    Potential energy surface</code>
  </div>
</div>

The event is added to the reference table only if it satisfies all the following conditions:

- $dE_{foward}$ < `emax_event`
- $dE_{foward}$ > `emin_event`
- $dE_{backward}$ > `emin_event`
- $dE_{backward}$ < `energy_asymmetry`x`backward_min` and $dE_{backward}$ > `backward_min`

Once a reference event is reused, it is refined to adapt to the current atomic configuration. The refinement is considered successful if:
- The central atom moves less than `refined_minimum_delr_thr` between the current position and the refined minimum.
- The difference in energy barriers between the generic and refined events is less than `refined_energy_thr`.
These thresholds ensure that the refined event remains consistent with the original.

To further control the behavior of the selected event search algorithm (style), you must define a separate section matching the algorithm name. For instance, if you choose `style = partn` you must also include a `[pARTn]` section in your INI file to configure specific parameters for the pARTn method.

### pARTn :

All other parameters have default values (see the full parameter documentation), but depending on your system, you may need to adjust some of them for optimal performance.
Parameters related to refinements are prefixed with r_

A minimal `[pARTn]` section will look like this :
```INI
[pARTn]
delr_thr = 0.1
r_eigval_th = -0.02
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
- $k_{b}$​ is the Boltzmann constant,
- $h$ is Planck’s constant.

All parameters must be provided in LAMMPS metal units.

Typically, it will gives :

```INI
[RateConstant]
style = constant
k0 = 10
T = 300
```

## PSR

When applying a reference event to a system for refinement, pyKMC performs a point set registration (also known as shape matching) to align the atomic positions from the reference event with the local atomic environment of the atom currently targeted for refinement.

You can define the algorithm to use with the `style` key. _Currently only 'ira' in implemented_.  This is the only mandatory parameter.

You may also want to adjust the default value of matching_score_thr, which sets the maximum allowed matching score for a successful registration.
For `style = ira`, the matching score corresponds to the Hausdorff distance between the two point sets.

This will gives you :
```INI
[PSR]
style = ira
matching_score_thr = 0.3
```

As with the Engine of EventSearch parts, you must also include a dedicated section for the selected point set registration method.

### IRA

When using, IRA, you can change default parameters that are being used, to have a full explaination please refer to the full parameters documentation and the IRA documentation.
To ensure access to default values, the section must be present in the configuration file, even if it's empty.

## Bias

Event selection normally draws from the active event table in proportion to each event's rate constant. A `[Bias]` section lets you steer that draw toward events that move the system in a direction you choose — for example pushing a surface atom into the bulk, or moving a vacancy toward another defect — while every other aspect of the KMC algorithm (rates, timestep, environment reuse) is unaffected.

To use it, set `bias = True` in `[Control]` and add a `[Bias]` section. Every `[Bias]` section needs a `style` key, which selects the geometric test applied to each candidate event, and a `mode` key, which selects how accepted events affect selection. The remaining keys depend on `style` and `mode`, as detailed below. For the complete list of `[Bias]` keys and their defaults, see the full parameter documentation.

### `style = direction`

Accepts events where the moving atom's displacement projects onto a fixed `direction` vector by at least `threshold`. Applies to every atom in the system unless restricted with `atom_indices` (see below).

```INI
[Bias]
style = direction
mode = filter
direction = [0.0, 0.0, -1.0]
threshold = 0.0
```

### `style = point`

Accepts events where the displacement projects, by at least `threshold`, onto the direction from the atom's current position toward a fixed `target_point`. Unlike `direction`, the reference direction is recomputed for each atom at each step, so it also applies to every atom in the system unless restricted with `atom_indices`.

```INI
[Bias]
style = point
mode = filter
target_point = [14.08, 14.08, 0.0]
threshold = 0.0
```

### `style = topo`

Accepts events where the moving atom carries the topology of a chosen defect. Set `atom_source_idx` to an atom carrying that defect at the start of the simulation — pyKMC then tracks the defect's topology ID rather than any single atom, so the bias keeps following the defect as it moves from one atom to another. `atom_indices`, `require_center` and `pass_unlisted` do not apply to this style: only atoms currently carrying the source topology are ever considered. Two variants are available.

Two-index mode: also set `atom_target_idx` to an atom carrying a second, target defect's topology. Events are accepted when they reduce the distance between the source and target defects; if the target topology disappears from the system on a given step, the bias is inactive for that step.

```INI
[Bias]
style = topo
mode = filter
atom_source_idx = 31
atom_target_idx = 108
```

One-index mode: omit `atom_target_idx` and instead give a `direction` and `threshold`, as in `style = direction`.

```INI
[Bias]
style = topo
mode = filter
atom_source_idx = 31
direction = [0.0, 0.0, 1.0]
threshold = 0.1
```

### Restricting to specific atoms

For `direction` and `point`, `atom_indices` restricts the bias to a subset of atoms; `require_center` and `pass_unlisted` then control how that subset is tested. With `require_center = False` (default), an event is accepted if any listed atom in its neighbourhood satisfies the condition. With `require_center = True`, only the event's central (moving) atom is tested — the example below accepts only events whose central atom is atom 0, moving downward:

```INI
[Bias]
style = direction
mode = filter
atom_indices = [0]
require_center = True
direction = [0.0, 0.0, -1.0]
threshold = 0.0
```

By default (`pass_unlisted = False`), an event whose atoms are all outside `atom_indices` is rejected by the bias. Setting `pass_unlisted = True` instead lets such events through unconditionally — only atoms in `atom_indices` are actually tested, everything else is free to proceed. This is only valid with `mode = filter`:

```INI
[Bias]
style = direction
mode = filter
atom_indices = [0, 1, 2]
pass_unlisted = True
direction = [0.0, 0.0, -1.0]
threshold = 0.0
```

### `mode = filter`

The default mode: a hard constraint. Events are drawn one at a time in proportion to their rate; any draw that the bias rejects is discarded and the draw repeats among the remaining events. If every event is eventually rejected, selection falls back to the unbiased draw over all events. The examples above all use this mode.

### `mode = boost`

A soft constraint. Rates of the bias-accepted events are multiplied by a common factor so that, together, they are selected with probability `bias_weight` at each step; non-accepted events keep their true rates and remain eligible. If the accepted events already reach `bias_weight` under their true rates, no boosting is applied. The reported `delta_t` is always corrected back to the true (unboosted) total rate, so simulation time remains physical. `pass_unlisted` cannot be used in this mode.

```INI
[Bias]
style = direction
mode = boost
atom_indices = [0]
direction = [0.0, 0.0, -1.0]
threshold = 0.0
bias_weight = 0.4
```

`thr_boost` excludes any accepted event whose `energy_barrier` (in eV) exceeds it from the boost, leaving its rate unmodified — useful to keep the boost from amplifying an unusually high barrier:

```INI
[Bias]
style = direction
mode = boost
atom_indices = [0]
direction = [0.0, 0.0, -1.0]
threshold = 0.0
bias_weight = 0.4
thr_boost = 1.0
```

Working, tested examples are available under `examples/NiSi_surface` (`style = direction`, `mode = boost`) and `examples/Si-vac-bias` (`style = topo`, one-index mode).

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
r_eigval_th = -0.02

[RateConstant]
style = constant
k0 = 1e12
T = 300

[PSR]
style = ira
matching_score_thr = 0.3

[IRA]
```

## Running a simulation

Once both the input file and the initial configuration file are ready, launch the simulation by executing:

```bash
python -m pykmc -in <your_input_file_name>
```

## Using multiple lammps instances

To speed up event searches and the refinement part of the KMC loop, it is possible to use multiple LAMMPS instances.
Each instance runs independently on a separate set of cores.

The number of instances is defined by the n_sessions parameter in the [Control] section of the input file.
The main KMC loop always runs on the main rank (rank 0), but you can choose whether the LAMMPS instances should use all other ranks only, or include rank 0 as well.
This behavior is controlled by the engine_use_rank_0 parameter in the [Control] section.
Note that enabling the use of rank 0 may slightly slow down the simulation, since that instance communicates via threads rather than MPI messages.

For example, if you have 8 cores available and want to run 4 LAMMPS instances, each on different ranks than the master one, your input file should contain the following parameters:

```INI
[Control]
...
n_sessions = 4
engine_use_rank_0 = False
...
```

You can then start the simulation with:
```bash
mpirun -n 8 python -m pykmc -in <your_input_file>
```
The available cores will be automatically split according to your configuration.
In this example, the LAMMPS instances will run on ranks 1–2, 3–4, 5–6, and 7, while the main KMC loop runs on rank 0.















