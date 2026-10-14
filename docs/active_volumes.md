# Active Volumes

Active Volume (AV) settings are defined in the `[ActiveVolume]` section of the input file.
The required parameters are ract and rmov for the total radius of the AV and the movable core radius, 
respectively
You must also enable AV mode in the `[Control]` section.

Example:

```INI
[Control]
...
active_volume = True
...

[ActiveVolume]
ract = 21
rmov = 15
```

*Note: ract must be larger than or equal to rmov to work. The difference between the two is the buffer region of the AV i.e. `ract`-`rmov`=buffer*

---

## General Idea

In very large systems with many defects, it can become memory intensive and slow to run event searches on the entire system. 
AV's define a region around a defect where searches and refinements will be performed without needing to use the entire system.
The AV has two parameters, the active radius `ract` and the movable radius `rmov`. The active radius encompasses all the atoms 
that are included in the AV, and captures the state of the overall system. The movable radius is where atoms are allowed to 
be moved during event searches and refinement. A frozen buffer region is made on the exterior of the AV in order to accurately model
the forces between atoms near the boundary.

FIGURE of AV

## Algorithm
1. **Reset LAMMPS Instance**: LAMMPS is cleared to prevent carry over of previous AV conditions
2. **AV Definition**: Find which atoms are to be included in AV, and which are movable
3. **Atom Map**: Create map between full system and AV system atom index's, types and positions
4. **AV Creation**: Atoms are created within AV

This then branches depending on if it is an Event Search step or Refinement step

**Event Search**: Event search continues the same as without AV's. Once results are obtained, the positions are mapped 
back to those of the full system.

**Refinement**: Unlike a non-AV refinement, the positions sent for refinement are not at the saddle point. First, the 
system in its current state is sent, the AV defined, then the potential energy `E_init` is calculated. This is needed as
regular refinements compare the saddle point energy to that of the full system, however in the AV the saddle point 
energy is in reference to that within the AV, not the full system. The event being refined is then applied to the atoms
saved in the local atomic environment, then the system is refined. The refined activation energy is then calculated from
`E_saddle - E_init` and the positions mapped back to the full system. 


If the movable radius `rmov` is less than the cut-off radius `rcut` for the local atomic environment, the refinements 
will fail. Similarly, if the active volume radius `ract` is less than `rcut`, the Event Searches will fail.

A debug mode to check if the AV is large enough can be toggled in `[ActiveVolume]` by setting `AV_debug = True`. This
will minimize the AV during refinement before the event is applied, and compare the energy before and after. The system 
being sent for refinement is already minimized, so there should be no difference after minimization within the AV, 
within 10e-3 eV. If there is a difference, it means the settings for the AV need to be adjusted. This is caused by
the buffer region being less than the cutoff of the potential being used, leading to movable atoms next to the buffer to
experience improper forces. This can be fixed by either increasing `ract` or decreasing `rmov`

