# Glossary

Use this page as a lookup while reading the examples.

## KMC

**Kinetic Monte Carlo** selects discrete transitions according to their rates and advances a simulated clock. A KMC step is an event selection, rather than a fixed-duration molecular-dynamics timestep. A successful basin exit can summarize several internal transitions.

## ARTn and pARTn

**Activation–Relaxation Technique nouveau (ARTn)** searches for saddle points and connected minima on a potential-energy surface. **pARTn** is the implementation used by these examples through LAMMPS. Several search attempts can fail or find equivalent events. See the [method references](references.md).

## CNA

**Common Neighbor Analysis** classifies local atomic structure using neighbor relationships. The `cna/graph` setting combines structural classification with graph-based environment identification. Its neighbor radius `rnei` describes recognition geometry; the potential has its own force cutoff. See the [CNA manual](https://docs.ovito.org/reference/pipelines/modifiers/common_neighbor_analysis.html).

## PSR and IRA

**Point Set Registration (PSR)** aligns atomic coordinate sets to test whether an environment matches a stored event. **Iterative Rotations and Assignments (IRA)** is the matching library used here. Graph identification provides candidate matches; geometric registration checks their actual shapes. See the [pyKMC parameter documentation](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/docs/parameters_details.md).

## Vacancy, SIA and dumbbell

A **vacancy** is an unoccupied lattice site. A **self-interstitial atom (SIA)** is an excess atom of the host material. A **dumbbell** is an arrangement in which two nearby atoms straddle a lattice site; together they accommodate one extra atom. A vacancy and an SIA form a **Frenkel pair**. Their recombination removes both defects while conserving atom count.

## Wigner–Seitz analysis

**Wigner–Seitz (WS)** analysis assigns atoms to sites of a reference crystal. Occupancy 0 marks a vacancy; occupancy above 1 contributes excess atoms. OVITO's **Sites** output displays reference sites, while **Atoms** displays the actual atoms. See the [WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html).

## Basin and flickering

**Flickering** is repeated low-barrier motion between nearby minima with little long-range progress. A **basin** groups connected states so acceleration can replace repeated internal transitions with an exit and residence time. The energy threshold controls grouping; it is distinct from the force tolerance used in searches or minimization. See the [Fe explanation](examples/fe-bivacancy/README.md#basins-and-flickering).

## Active volume

An **active volume (AV)** limits a local calculation to a region around an event. `rmov` is the radius of movable atoms; `ract` is the radius of the entire spherical region. The shell between them is a frozen buffer. These lengths are in Å and serve a different purpose from the environment-recognition radius `rcut`. See the [parameter definitions](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/docs/parameters_details.md#activevolume-section-optional).

## Reference event and active event

A **reference event** stores a reusable transition. An **active event** is a candidate mapped to the current configuration. Symmetry and multiple possible locations can produce several active events from one reference. A candidate becomes an executed event only when selected and successfully applied.

## Rate and barrier

The forward barrier is the energy difference between the initial minimum and saddle. The reverse barrier is measured from the final minimum to that saddle. With a constant prefactor, the rate is `k = k0 exp(-Ea/kBT)`. It has units of inverse time, not velocity. See the [rate and output explanation](examples/ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time).

## MPI process, rank, task and session

**MPI** means Message Passing Interface. An MPI **process** runs part of the program; its **rank** is its identifier within a communicator. A Slurm **task** is a launched process slot, and these job scripts use one MPI process per task.

A pyKMC **session** is an engine worker managed by the controller. These examples launch 8 MPI processes: rank 0 controls the calculation, and 7 worker sessions perform engine work. More complex group layouts are possible; process and session counts are not universally interchangeable.
