## `Control` Section (mandatory)

<details><summary>Section Overview</summary>
  Core simulation control parameters.
</details>

- **`initial_config`** : `str`, mandatory
  <details><summary>Description</summary>
  File path for the initial atomic structure. This file should be parseable by `ase.io.read()` and contain atom types, positions, simulation cell, and periodic boundary conditions.
  </details>
- **`trajectory_output`** : `str`, default = `'./trajkmc.xyz'`
  <details><summary>Description</summary>
  File path where the simulation trajectory will be saved. The file should be writable by `ase.io.write` using `append=True `
  </details>
- **`reference_table_output`** : `str`, default = `'./reference_table.pickle'`
  <details><summary>Description</summary>
  File path where the reference table will be store in pickle format.
  </details>
- **`visited_environments_output`** : `str`, default = `'./visited_environments.pickle'`
  <details><summary>Description</summary>
  File path where the list of atomic environments that have been explored will be sore in pickle format.
  </details>
- **`reference_table`** : `str`, optional
  <details><summary>Description</summary>
  Path to a reference table generated from a previous simulation.
  </details>
- **`visited_environments`** : `str`, optional
  <details><summary>Description</summary>
  Path to a list of visited environment generated from a previous simulation.
  </details>
- **`restart_file`** : `str`, optional
  <details><summary>Description</summary>
  File with restart informations.
  </details>
- **`reconstruction`** : `bool`, default = `True`
  <details><summary>Description</summary>
  If at each KMC step we reconstruct generic events.
   NOT WORKING
  </details>
- **`n_steps`** : `int`, mandatory
  <details><summary>Description</summary>
  Total number of simulation steps to run.
  </details>
- **`engine`** : `Literal['lammps']`, mandatory
  <details><summary>Description</summary>
  Which E/F Engine to use. Note : Only lammps is implemented.
  </details>
- **`n_sessions`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of Sessions
  </details>
- **`engine_use_rank_0`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Deprecated : If use mpi rank 0 or not.
  </details>
- **`verbosity`** : `int`, default = `1`
  <details><summary>Description</summary>
  Controls the level of detail in the simulation output.
  </details>
- **`refine_thr`** : `float`, default = `0.9999`
  <details><summary>Description</summary>
  Event constributing to this percent of ktot are refined.
  </details>
- **`basin`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Basin mode
  </details>
- **`active_volume`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Incorporate AV's into simulations, recommended for large systems
  </details>
- **`recycle`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Recycle non-perturbed events from the previous KMC step instead of re-searching them. Requires an [EventRecycling] section.
  </details>
- **`bias`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Enable event selection bias. Requires a [Bias] section.
  </details>

---

## `Atomicenvironment` Section (mandatory)

<details><summary>Section Overview</summary>
  Atomic environments parameters.
</details>

- **`style`** : `Literal['cna', 'graph', 'cna/graph', 'coordination', 'coordination/graph', 'diamond/graph']`, mandatory
  <details><summary>Description</summary>
  Method used to characterize and assign an ID to an atom's local atomic environment. 'coordination' classifies atoms based on nearest-neighbor count against a threshold. 'coordination/graph' first filters by coordination, then computes graph IDs for non-crystal atoms.
  </details>
- **`rnei`** : `float`, mandatory
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors.
  </details>
- **`rcut`** : `float`, optional
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the local atomic environment.
  </details>
- **`neighbors_add`** : `int`, default = `0`
  <details><summary>Description</summary>
  When `style` is 'cna/graph', specifies the N-th shell of neighbors whose graph IDs should also be computed.
  </details>
- **`coordination_threshold`** : `int`, optional
  <details><summary>Description</summary>
  When style is 'coordination' or 'coordination/graph', atoms with fewer neighbors (within rnei) than this value are classified as 'noncrystal'. Atoms with this many or more neighbors are classified as 'crystal'. Required when style is 'coordination' or 'coordination/graph'.
  </details>
- **`atom_coloring_mode`** : `Literal['grey', 'full']`, default = `'grey'`
  <details><summary>Description</summary>
  Controls whether element types are used in environment matching. 'grey': all atoms treated identically (grey alloy approximation). 'full': element types used in graph hashing, PSR matching, and symmetry detection.
  </details>

---

## `Eventsearch` Section (mandatory)

<details><summary>Section Overview</summary>
  Event search parameters.
</details>

- **`style`** : `Literal['partn']`, mandatory
  <details><summary>Description</summary>
  Method used to find events.
  </details>
- **`nsearch`** : `int`, mandatory
  <details><summary>Description</summary>
  Number of event searches to perform per unique atomic environment.
  </details>
- **`emax_event`** : `float`, default = `5.0`
  <details><summary>Description</summary>
  Maximum energy barrier (in eV) for an event to be added to the reference table.
  </details>
- **`emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum energy forward and backward barrier (in eV) for an event to be added to the reference table.
  </details>
- **`backward_emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  To be used with `energy_assymetry`.
  </details>
- **`energy_asymmetry`** : `int`, default = `5`
  <details><summary>Description</summary>
  Prevent highly asymmetric event to be added to the reference table.The con
  </details>
- **`refined_minimum_delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Refinement is accepted only if the central atom moves less than this distance between the current position and the refined minimum.
  </details>
- **`refined_energy_thr`** : `float`, default = `0.05`
  <details><summary>Description</summary>
  Maximumallowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy.
  </details>
- **`delr_thr`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  delr threshold between one minima and the intial configuration to consider the event valid.
  </details>

---

## `Psr` Section (mandatory)

<details><summary>Section Overview</summary>
  Point set registration parameters.
</details>

- **`style`** : `Literal['ira']`, mandatory
  <details><summary>Description</summary>
  Method used for the point set registration (shape matching) between reference events and atomic environment of an atom having the same atomic environement ID of the event. This method is also used to find atomic environment symmetries.
  </details>
- **`matching_score_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Maximum value of the matching score of the algorithm used.
  </details>

---

## `Rateconstant` Section (mandatory)

<details><summary>Section Overview</summary>
  Rate constant computation parameters.
</details>

- **`style`** : `Literal['constant']`, mandatory
  <details><summary>Description</summary>
  Method used to compute the prefactor of the rate constant. 
  </details>
- **`k0`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  When `style` is set to **'constant'**, this value is used directly as the pre-exponential factor ($k_0$) 
  $$ k = k_{0} \exp\left(-\frac{\Delta E}{k_{b}T}\right) $$
  </details>
- **`T`** : `float`, default = `300`
  <details><summary>Description</summary>
  Temperature (in Kelvin) used for computing rate constants.
  </details>

---

## `Lammps` Section (optional)

<details><summary>Section Overview</summary>
  Lammps parameters.
</details>

- **`pair_style`** : `str`, mandatory
  <details><summary>Description</summary>
  Lammps pair_style command.
  </details>
- **`pair_coeff`** : `str`, mandatory
  <details><summary>Description</summary>
  Lammps pair_coeff command.
  </details>
- **`min_style`** : `str`, default = `'cg'`
  <details><summary>Description</summary>
  Lammps min_style command.
  </details>
- **`minimize`** : `str`, default = `'1.0e-6 1.0e-8 1000 1000'`
  <details><summary>Description</summary>
  Lammps minimize command
  </details>
- **`boundary`** : `str`, optional
  <details><summary>Description</summary>
  LAMMPS boundary string (e.g. 'p p p' or 'p p f'). If None, derived from the input structure's PBC flags.
  </details>

---

## `Partn` Section (optional)

<details><summary>Section Overview</summary>
  pARTn parameters.
</details>

- **`verbosity`** : `int`, default = `2`
  <details><summary>Description</summary>
  pARTn verbosity
  </details>
- **`delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Threshold at which an atom is considered to have moved. This threshold affects the npart parameter in the artn.out output.
  </details>
- **`zseed`** : `int`, default = `0`
  <details><summary>Description</summary>
  The value of zseed is used to seed the random number generator. If the value equals 0, a new radom seed gets geenrated. The exact zseed value of each research is written in file zseed.dat, which can be useful for debugging, or re-running exact same pARTn runs.
  </details>
- **`push_mode`** : `Literal['list', 'rad']`, default = `'rad'`
  <details><summary>Description</summary>
  Determines how the initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) from the central atom.
  </details>
- **`push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which all atoms receive an initial displacement.
  </details>
- **`push_step_size`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- **`ninit`** : `int`, default = `2`
  <details><summary>Description</summary>
  Specify the minimal number of pushes with the initial push vector.
  </details>
- **`lanczos_min_size`** : `int`, default = `10`
  <details><summary>Description</summary>
  Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`lanczos_max_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Maximum number of Lanczos iterations.
  </details>
- **`lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- **`eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- **`eigen_step_size`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  The limit to the maximum size of the displacement with eigenvector.
  </details>
- **`nsmooth`** : `int`, default = `3`
  <details><summary>Description</summary>
  Number of smoothing steps from initial displacement to eigenvector.
  </details>
- **`neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of pushes along the eignevector before starting a perpendicular relax.
  </details>
- **`alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- **`nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Number of times a research is allowed to cross a convex region (without counting the starting convex region).
  </details>
- **`nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Control the perpendicular relaxation.
  </details>
- **`nperp_limitation`** : `list[int]`, default = `[4, 8, 12, 16, -1]`
  <details><summary>Description</summary>
  Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- **`forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- **`convergence_property`** : `Literal['maxval', 'norm']`, default = `'maxval'`
  <details><summary>Description</summary>
  Specify how to test convergence of the forces. 'maxval': the convergence will be tested by MAXVAL( ABS( force ) ); 'norm' the convergence will be tested by NORM2( force ).
  </details>
- **`nevalf_max`** : `int`, default = `9999`
  <details><summary>Description</summary>
  Stop an artn search before end when the number of force evaluations by the force engine is greater to nevalf_max
  </details>
- **`push_over`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Factor that scales the displacement vector used to push the system from the saddle point towards a local energy minimum. 
  $$ \text{displacement} = \text{push_factor} \times v_0 \times \text{eigen_step_size} \times \text{push_over} \times 0.8 $$
  </details>
- **`dmax`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.
  </details>
- **`r_nevalf_max`** : `int`, default = `300`
  <details><summary>Description</summary>
  Stop an artn refinement before end when the number of force evaluations by the force engine is greater to nevalf_max.
  </details>
- **`r_max_attempts`** : `int`, default = `5`
  <details><summary>Description</summary>
  When adjusting the saddle energy and positions, in some rare cases partn has trouble finding the saddle point and goes back to the minium.In that case, we do another attempt with a different seed.
  </details>
- **`r_delr_sad_thr`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  When a saddle point is found by pARTn, we compare artn delr_sad to this threshold to check if the system went back to the minimum. If yes, new attempt.
  </details>
- **`r_push_mode`** : `Literal['list', 'rad']`, default = `'list'`
  <details><summary>Description</summary>
  Determines how the initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) from the central atom.
  </details>
- **`r_push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which all atoms receive an initial displacement.
  </details>
- **`r_push_step_size`** : `float`, default = `0.0001`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- **`r_ninit`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Specify the minimal number of pushes with the initial push vector.
  </details>
- **`r_lanczos_min_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Refinement: Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`r_lanczos_max_size`** : `int`, default = `50`
  <details><summary>Description</summary>
  Refinement: Maximum number of Lanczos iterations.
  </details>
- **`r_lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Refinement: Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`r_lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- **`r_eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Refinement: threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- **`r_eigen_step_size`** : `float`, default = `0.005`
  <details><summary>Description</summary>
  Refinement: The limit to the maximum size of the displacement with eigenvector.
  </details>
- **`r_nsmooth`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of smoothing steps from initial displacement to eigenvector.
  </details>
- **`r_neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Refinement: Number of pushes along the eignevector before starting a perpendicular relax.
  </details>
- **`r_alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  Refinement: This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- **`r_nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of times a research is allowed to cross a convex region (without counting the starting convex region).
  </details>
- **`r_nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Refinement: Control the perpendicular relaxation.
  </details>
- **`r_nperp_limitation`** : `list[int]`, default = `[100]`
  <details><summary>Description</summary>
  Refinement: Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- **`r_forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Refinement: The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- **`r_dmax`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Refinement: dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.
  </details>

---

## `Ira` Section (optional)

<details><summary>Section Overview</summary>
  IRA parameters.
</details>

- **`kmax_factor`** : `float`, default = `1.8`
  <details><summary>Description</summary>
  Multiplicative factor that needs to be larger than 1.0. Larger value increases the search space of the rotations.
  </details>
- **`sym_thr`** : `float`, default = `0.01`
  <details><summary>Description</summary>
  Threshold in terms of the Hausdorff distance. If an operation returns a distance value beyond sym_thr, then SOFI will not consider that operation as a symmetry operation.
  </details>

---

## `Basin` Section (optional)

<details><summary>Section Overview</summary>
  Basin parameters
</details>

- **`style`** : `Literal['global', 'global/reconstruction']`, default = `'global'`
  <details><summary>Description</summary>
  Basin style used.
  </details>
- **`energy_thr`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Energy threshold
  </details>

---

## `Reconstruction` Section (mandatory)

<details><summary>Section Overview</summary>
  Reconstruction parameters.
</details>

- **`push_fraction`** : `float`, default = `0.15`
  <details><summary>Description</summary>
  Fraction used to push the system from the saddle point toward each minimum during reconstruction.
  </details>

---

## `Activevolume` Section (optional)

<details><summary>Section Overview</summary>
  Active Volume Parameters
</details>

- **`ract`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  Radius of entire active volume, spherical
  </details>
- **`rmov`** : `float`, default = `4.0`
  <details><summary>Description</summary>
  Radius of movable atoms in active volume, spherical
  </details>
- **`AV_debug`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Debug flag for active volume size checks
  </details>

---

## `Dealloying` Section (optional)

<details><summary>Section Overview</summary>
  Dealloying event parameters.
  
  When enabled, atoms with coordination number below the threshold
  are eligible for removal via a fixed-rate dealloying event.
</details>

- **`coordination_threshold`** : `int`, mandatory
  <details><summary>Description</summary>
  Atoms with fewer neighbors (within rnei) than this value are eligible for dealloying.
  </details>
- **`rate_constant`** : `float`, mandatory
  <details><summary>Description</summary>
  Fixed rate constant (ps^-1) for dealloying events.
  </details>
- **`eligible_types`** : `list[str]`, optional
  <details><summary>Description</summary>
  If provided, only atoms of these element types are eligible for dealloying. If None, all under-coordinated atoms are eligible.
  </details>

---

## `Eventrecycling` Section (optional)

<details><summary>Section Overview</summary>
  Event recycling parameters. Required when control.recycle = True.
</details>

- **`style`** : `Literal['displacement']`, mandatory
  <details><summary>Description</summary>
  Method used to decide which events can be recycled. 'displacement' = central atom moved less than movement_thr AND is farther than distance_thr from the executed event.
  </details>
- **`movement_thr`** : `float`, default = `0.02`
  <details><summary>Description</summary>
  Angstroms. Central atoms whose displacement from pre- to post-execution is below this are considered 'unmoved'.
  </details>
- **`distance_thr`** : `float`, default = `10.0`
  <details><summary>Description</summary>
  Angstroms. Candidate events whose central atom is farther than this (PBC-aware minimum-image) from the executed event's central atom pass the distance check.
  </details>

---

## `Inactive_atoms` Section (optional)

<details><summary>Section Overview</summary>
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/region.py``.
</details>

- **`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- **`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- **`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- **`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- **`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- **`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- **`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- **`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- **`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- **`types`** : `list[str]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- **`indices`** : `list[int]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

## `Frozen_atoms` Section (optional)

<details><summary>Section Overview</summary>
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/region.py``.
</details>

- **`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- **`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- **`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- **`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- **`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- **`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- **`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- **`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- **`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- **`types`** : `list[str]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- **`indices`** : `list[int]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

## `Bias` Section (optional)

<details><summary>Section Overview</summary>
  Event selection bias parameters.
</details>

- **`style`** : `Literal['direction', 'point', 'topo']`, mandatory
  <details><summary>Description</summary>
  Bias style: 'direction' (DirectionBias), 'point' (PointBias), or 'topo' (TopoBias).
  </details>
- **`mode`** : `Literal['filter', 'boost']`, default = `'filter'`
  <details><summary>Description</summary>
  Selection mode. 'filter': rejection-loop removes non-accepted events. 'boost': multiplies desired event rates by a dynamic factor so they fire with probability bias_weight, without blocking other events.
  </details>
- **`bias_weight`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  Target probability in (0, 1) that a desired event is selected at each step. Only used in boost mode.
  </details>
- **`pass_unlisted`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Whether atoms not in atom_indices pass through the bias predicate unchanged. False (default): non-listed atoms are rejected/undesired. True: non-listed atoms always pass; only valid in filter mode.
  </details>
- **`direction`** : `list[float]`, optional
  <details><summary>Description</summary>
  Direction vector [x, y, z] for 'direction' bias.
  </details>
- **`target_point`** : `list[float]`, optional
  <details><summary>Description</summary>
  Target point [x, y, z] for 'point' bias.
  </details>
- **`atom_indices`** : `list[int]`, optional
  <details><summary>Description</summary>
  Global atom indices to bias. None means all atoms.
  </details>
- **`threshold`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum projection onto the bias direction for acceptance.
  </details>
- **`topo_source`** : `str`, optional
  <details><summary>Description</summary>
  Source topology ID for 'topo' bias (e.g. vacancy).
  </details>
- **`topo_target`** : `str`, optional
  <details><summary>Description</summary>
  Target topology ID for 'topo' bias (e.g. interstitial).
  </details>

---
