# HTST rate prefactors

With `[RateConstant] style = htst`, pyKMC computes a harmonic transition state theory (HTST)
prefactor $\nu_0$ per reference event instead of using the same `k0` for all of them.
[Which events](#which-events-get-a-prefactor-and-what-it-costs) use which $\nu_0$ is
described below.

Every number on this page was measured on a first-neighbour vacancy hop in fcc Ni with the
Ni v6 EAM potential (`tests/data/Ni_v6_2.0_LKBeland2016.eam`), unless stated otherwise.

## What it does

The rate constant of an event is

$$
k = \nu_{0} e^{-\frac{dE_{forward}}{k_{b}T}}
$$

where $\nu_0$ is the Vineyard prefactor (Vineyard, 1957): the product of the $3n$ normal-mode
frequencies at the minimum divided by the product of the $3n-1$ real frequencies at the saddle,
for the $n$ atoms that are displaced to build the Hessians (with `free_radius = full` the three
translations are removed, which leaves $3n-3$ and $3n-4$ modes).

$$
\nu_{0} = \frac{\prod_{i=1}^{3n} \nu_{i}^{min}}{\prod_{i=1}^{3n-1} \nu_{i}^{saddle}}
$$

The Hessians are computed by finite differences with the LAMMPS `dynamical_matrix` command.

**Units.** Rate constants are in ps⁻¹. $\nu_0$ is in THz (1 THz = 1 ps⁻¹), the unit LAMMPS
reports frequencies in and the unit of `k0`. `k0` is in ps⁻¹: `k0 = 10` is $10^{13}$ Hz. Under
`style = htst`, `k0` is the prefactor of every reference event whose $\nu_0$ could not be
computed, and a `k0` above 1000 ps⁻¹ is refused at startup: a value typed in Hz by mistake
would make every such event dominate the simulation.

**Requirements.** LAMMPS built with the PHONON package, and an orthogonal simulation cell.

## Quick start

```ini
[RateConstant]
style = htst
k0 = 10
T = 300
```

With `active_volume = False`, the Hessian of an event is computed on the atoms within
`free_radius = 8.0` Å of the moving atom, surrounded by a frozen shell. With
`active_volume = True`, it is computed on a region with the radii of the active volume. See
[Hessian region](#hessian-region) and [Choosing the radius](#choosing-the-radius) before a
production run.

## Which events get a prefactor, and what it costs

| Event | Hessians | $\nu_0$ |
|---|---|---|
| New reference event | 3 (first minimum, saddle, second minimum), in one job | forward and backward |
| New reference event without a backward event of its own (it is its own reverse, or its reverse is already known) | 2 (first minimum, saddle) | forward |
| Active event, refined or not | none | the $\nu_0$ of its reference event, with its own energy barrier |
| Recycled active event | none | kept, like its refined energy barrier |
| Event inside a basin | none | the $\nu_0$ of its reference event (see [Basins](#basins)) |

- A reference event whose Hessian calculation fails uses `k0`, and a warning in `pykmc.log`
  names the event and the reason (see [Per-event fallbacks](#per-event-fallbacks)). The
  active events built on it use `k0` too.
- With a prefactor per event, the rate constant is no longer a monotone function of the energy
  barrier. The events that are not worth refining (`refine_thr`) are therefore selected by the
  rate constant of their reference event, not by its energy barrier, under every `style`.
  With a constant prefactor the two rules select the same events.
- Each Hessian takes $6 n$ force evaluations on the atoms present in the calculation, on one
  core of a session. Measured on one core of an Apple M1 Max:

| Region | Displaced / present atoms | Time per Hessian |
|---|---|---|
| Active volume, `rmov = 9`, `ract = 16` | 316 / 1552 | 3.4 s |
| `free_radius = 8` on a 2047-atom cell | 193 / 2019 | 3.1 s |
| `free_radius = full` on a 2047-atom cell | 2047 / 2047 | 67 s |

Jobs are spread over the sessions like event searches and refinements: one job per new
reference event.

## Hessian region

The atoms **displaced** by the finite differences define the Hessian. The other atoms **present**
in the calculation are kept fixed, and the rest of the system is left out.

| Setting | Displaced | Present and fixed | Left out |
|---|---|---|---|
| `active_volume = True` | within `rmov` of the moving atom's position at the minimum | the `rmov`–`ract` buffer | beyond `ract` |
| `active_volume = False` (default) | within `free_radius` of the moving atom's position at the saddle | a shell of two force cutoffs (`shell_width` overrides it) | beyond the shell |
| `active_volume = False`, `free_radius = full` | every atom (the three translations are projected out) | none | none |

- With `active_volume = True` the Hessian region has the radii of the active volume and is
  periodic in every direction, like it: `free_radius`, `shell_width` and
  `max_full_hessian_mb` are ignored (`free_radius = full` is still refused with a
  `[frozen_atoms]` section). It is centred on the moving atom of the event. The active volume
  of the search is centred on the atom the search started from: the two regions are the same
  atoms only when that atom is the moving atom.
- The force cutoff is read from LAMMPS at startup. It is 6.5 Å for Ni v6, 5.3 Å for the
  Mendelev Fe potential and 3.77118 Å for Stillinger-Weber Si, so the default shells are 13,
  10.6 and 7.54236 Å. A potential whose real range exceeds two cutoffs (message-passing
  machine-learning potentials, screened MEAM) needs a larger `shell_width`.
- The region is the same for an event and its reverse: both are computed in one job.
- Each calculation runs in a serial LAMMPS instance of its own, on one core of a session. The
  engines used for the searches and refinements are not modified.

## Which value is the reference

The prefactor depends on which atoms are displaced. pyKMC takes as the reference the all-atom
value at fixed volume: every atom of the cell displaced, which is what `free_radius = full`
computes. The two cropped regions differ from it as follows. The all-atom value is 23.28 THz
on the 2047-atom cell. It was not computed on the 8787-atom cell, on which the active-volume
rows were measured: 23.28 THz stands for it there, within about 2 %.

| Region | Setting | $\nu_0$ (THz) | Against the all-atom value |
|---|---|---|---|
| Whole system | 2047 atoms | 23.28 | reference |
| Active volume | `rmov = 9`, `ract = 16` | 23.13 | −0.6 % |
| Active volume | `rmov = 8`, `ract = 21` | 22.73 | −2.3 % |
| Active volume | `rmov = 4`, `ract = 6` (the defaults) | 11.47 | −51 % |
| Sphere | `free_radius = 8`, 2047-atom cell | 26.73 | +15 % |
| Sphere | `free_radius` = 8 to 18, 8787-atom cell | 27.2 to 28.3 | +17 to +22 % |

- With the active volume, the saddle is refined against a buffer held at the positions of the
  minimum. The prefactor is then within about 2 % of the all-atom value for `rmov` ≥ 8 Å:
  −2.3 % at 8 Å, −0.6 % at 9 Å and about +2 % from 12 to 16 Å.
- With the sphere, the rest of the cell sits at the relaxed positions of each state. The
  prefactor converges with the radius to a limit about 20 % above the all-atom value, not to
  the all-atom value. This offset is measured; its origin (the relaxation volume of the saddle)
  is the likely explanation, not a derivation.
- If the offset matters, use `free_radius = full` (cells up to about 3700 atoms with the
  default `max_full_hessian_mb`) or `active_volume = True`. Prefactors computed with a sphere
  are consistent among the events of a run, but all of them carry the offset.

## Choosing the radius

### Active volume

Use `rmov` ≥ 8 Å and a buffer `ract − rmov` ≥ 4 Å. A warning is written at startup otherwise.

- `rmov`: the prefactor is about 50 % low at `rmov = 4` Å and about 2 % low at `rmov = 8` Å.
- Buffer: the error caused by the crop is at most 0.4 % for a buffer of 4 Å or more, up to
  2.2 % at 2–3 Å, and between −39 % and +12 % at 1 Å. The 4 Å figure depends on the range of
  the potential.

- Keep `rmov` ≥ `rcut` (`[AtomicEnvironment]`). The refinement of an event starts from the
  atoms within `rcut` placed at their estimated saddle positions; those of them that are in
  the fixed buffer stay there, so with `rmov` < `rcut` the buffer of a refined event is not
  the same at the minimum and at the saddle.

The event search also requires `ract` > `rcut` (see [Active volumes](active_volumes.md)).

### Sphere (`active_volume = False`)

Converge `free_radius` on your own system:

1. Write an input with `n_steps = 1` and your usual `n_sessions`.
2. Run it in separate directories with `free_radius` = 4, 6, 8, 10 and 12. Where possible the
   cell should be larger than 2 × (`free_radius` + shell), so that the displaced atoms do not
   interact with their own periodic images.
3. In each `reference_table.pickle`, find the event of interest by its energy barrier and
   read its prefactor:
   ```python
   import pandas as pd

   table = pd.read_pickle("reference_table.pickle")
   print(table[["idx_ref", "energy_barrier", "k_pref"]])
   ```
4. Take the largest radius as the reference. Choose the smallest radius whose $\nu_0$ is within
   5 % of it, at that radius and at every larger one.

This converges to the limit of the sphere (see [above](#which-value-is-the-reference)), not to
the all-atom value.

| System | `free_radius` (Å) | Displaced atoms | $\nu_0$ (THz) | Against 12 Å | Time per event (s) |
|---|---|---|---|---|---|
| Ni v6, fcc, 2047 atoms | 4 | 23 | 15.38 | −43 % | 1.0 |
| | 6 | 83 | 22.04 | −19 % | 3.9 |
| | 8 | 193 | 26.73 | −1.8 % | 9.4 |
| | 10 | 383 | 27.08 | −0.4 % | 19 |
| | 12 | 669 | 27.20 | reference | 35 |
| Mendelev Fe, bcc, 1999 atoms, first half of the vacancy hop (forward) | 4 | 18 | 68.0 | −64 % | 0.3 |
| | 6 | 73 | 141.5 | −24 % | 1.5 |
| | 8 | 182 | 168.8 | −9.5 % | 4.4 |
| | 10 | 358 | 183.6 | −1.5 % | 8.9 |
| | 12 | 612 | 186.5 | reference | 16 |

The time is for one reference event (three Hessians) on one core of an Apple M1 Max.

With the 5 % rule this gives **8 Å for fcc Ni and 10 Å for bcc Fe**. The default is 8 Å.

- Both cells are 28 Å wide, so the rows at 10 and 12 Å include the interaction of the displaced
  atoms with their periodic images.
- With the Mendelev potential the vacancy hop of bcc Fe has two barriers and an intermediate
  minimum. The table is for its first half, and its large forward $\nu_0$ is specific to it.
- The same convergence with the radius is reported for the vacancy in α-Fe by Lazauskas
  *et al.* (2014). Lefèvre López *et al.* (2024) report prefactors of the same order of
  magnitude for FeNiCr alloys (6.6 × 10¹² s⁻¹ on average).

### Whole system (`free_radius = full`)

One Hessian is a matrix of $(3N)^2$ doubles: $8 (3N)^2 / 10^6$ MB, 302 MB for 2047 atoms. Up to
`n_sessions` Hessians are computed at once. Each of them needs about twice that size in memory
and writes a scratch file of that size in `hessian_dir`; a temporary directory held in memory
fills quickly. The run stops at startup if one Hessian is larger than `max_full_hessian_mb`
(1000 MB by default); raise it knowingly.

`free_radius = full` needs at least two periodic directions.

A body isolated by vacuum (a cluster, a molecule) gets no prefactor when all its atoms are
displaced, with `free_radius = full` or with a sphere that covers it entirely: its rotations
are zero modes, and they are not removed. Its events use `k0`, and the warning says that the
displaced atoms form an isolated body.

## Finite-difference step

`fd_step` is the displacement used by `dynamical_matrix` (0.01 Å by default). Do not go below
0.01 Å with tabulated (spline) EAM potentials. With the minimum and the saddle of the whole
8787-atom cell cropped to `rmov = 9`, `ract = 16`, the prefactor is 26.67 THz at 0.02 Å,
26.65 THz at 0.01 Å, 26.76 THz at 0.005 Å (+0.4 %) and 26.94 THz at 0.0025 Å (+1.1 %).

## Keys

All the keys are in the `[RateConstant]` section. See [KMC Parameters](parameters.md) for the
full descriptions.

| Key | Default | Meaning |
|---|---|---|
| `style` | mandatory | `constant` or `htst` |
| `k0` | `1.0` | prefactor (ps⁻¹) of the events without a $\nu_0$; at most 1000 under `htst` |
| `T` | `300` | temperature (K) |
| `free_radius` | `8.0` | radius (Å) of the sphere of displaced atoms, or `full`; ignored with the active volume |
| `shell_width` | two force cutoffs | thickness (Å) of the fixed shell around the sphere |
| `hessian_dir` | system temporary directory | directory of the Hessian scratch files |
| `fd_step` | `0.01` | finite-difference displacement (Å) |
| `max_full_hessian_mb` | `1000.0` | largest Hessian (MB) allowed with `free_radius = full` |

## Startup checks

A run that could never compute a prefactor stops at startup. The reason is written in
`pykmc.log` (configuration errors are printed by every MPI rank instead).

| Condition | What to do |
|---|---|
| `k0` above 1000 ps⁻¹ | give `k0` in ps⁻¹ (10 ps⁻¹ = $10^{13}$ Hz) |
| `free_radius` is neither a positive number nor `full` (under any `style`) | give a radius in Å, or `full` |
| A removed key of the `SK_HTST_backend` branch (`premin`, `nu0_zone_radius`, `require_one_negative_mode`, `nu0_min_THz`, `nu0_max_THz`), or `style = rpa` (under any `style`) | remove the key, or use `style = htst` for `rpa` (it computed the same prefactor); `DEPRECATED.md` on that branch maps every old setting |
| Non-orthogonal cell | use an orthogonal cell, or `style = constant` |
| `free_radius = full` with a `[frozen_atoms]` section (also with `active_volume = True`) | use a numeric `free_radius`, or remove `[frozen_atoms]` |
| `free_radius = full` with fewer than two periodic directions | use a numeric `free_radius` |
| `free_radius = full` and a Hessian above `max_full_hessian_mb` | raise the limit, or use a numeric `free_radius` |
| `active_volume = True` with `rmov` = `ract` (`rmov` > `ract` is refused under any `style`) | the Hessian needs the fixed buffer: use `rmov` < `ract` |
| `hessian_dir` cannot be created or written | give a writable directory |
| LAMMPS has no PHONON package | rebuild LAMMPS with `-DPKG_PHONON=on` |
| The potential fails on a test box with one atom per species | check `pair_style` and `pair_coeff` |
| `active_volume = False` with a numeric `free_radius` and no `shell_width`, and the force cutoff cannot be read from LAMMPS | set `shell_width` |

Warnings are written, and the run continues, when:

- `active_volume = True` and `rmov` < 8 Å, or `ract − rmov` < 4 Å
  (see [Choosing the radius](#choosing-the-radius));
- `atom_coloring_mode = grey` with more than one species: the $\nu_0$ of a reference event is
  computed with the species of the event it was found on, and it is reused by the events on
  other species.

Unless `verbosity = 0`, the log also gives, at startup, the LAMMPS version, the scratch
directory, `fd_step` and the Hessian region.

## Per-event fallbacks

When the prefactor of a reference event cannot be computed, the event uses `k0` and a warning
is written:

```
WARNING : Rate constant compute failed on reference event 12 (minimum has 1 negative modes (expected 0)), use backup value k0 = 10 ps^-1
```

| Error code | Reason |
|---|---|
| `PrefactorError.SADDLE_NOT_FIRST_ORDER` | the saddle does not have exactly one negative mode |
| `PrefactorError.UNSTABLE_MINIMUM` | the minimum has a negative mode |
| `PrefactorError.NONFINITE_PREFACTOR` | $\nu_0$ is not a finite positive number (a zero mode, such as the rotations of an isolated body, or non-finite Hessian entries) |
| `PrefactorError.FROZEN_OVERLAP` | a frozen atom is among the displaced atoms (see [Frozen atoms](#frozen-atoms)) |
| `PrefactorError.WRAPPED_OPEN_AXIS` | the event geometry was wrapped across a non-periodic direction |
| `PrefactorError.NO_FROZEN_BOUNDARY` | every atom present is displaced, but the region is not the whole system |
| `PrefactorError.ENGINE_FAILURE` | the calculation failed (a LAMMPS error, atoms lost outside a non-periodic boundary, an incomplete Hessian file, or a job that could not be sent or collected); the message gives the error |

The reference table stores the prefactor of each event in `k_pref` (ps⁻¹, i.e. THz): its
$\nu_0$, or `k0` when the calculation failed.

- There is no acceptance window: a $\nu_0$ outside 0.1–1000 THz is used as computed, and a
  warning is written.
- There is no test on the residual forces of the minimum or the saddle: the number of negative
  modes and the finiteness of the result are the only checks. A loose `[Lammps] minimize`
  adds its own error to $\nu_0$.
- Unless `verbosity = 0`, the log gives, at each step that has some, the number of prefactors
  computed for the new reference events (one per direction).

## Frozen atoms

- If an atom of `[frozen_atoms]` is among the displaced atoms of an event, the event uses `k0`
  and the warning says that the frozen region overlapped the HTST calculation space.
- A frozen atom that is only in the fixed shell or buffer has no effect.
- `free_radius = full` cannot be used with `[frozen_atoms]`, whatever `active_volume` is.

## Basins

With `basin = True`, the rate constants inside a basin (the internal hops and the exits) use
the prefactor of their reference event, like the active events.

## Restarts and reused reference tables

A reference event stores only the atoms around the event, so its $\nu_0$ cannot be recomputed
when a reference table is loaded (`reference_table`). A loaded table is used as it was
written: its rate constants (`k`) and prefactors (`k_pref`) are kept, whatever the `style`,
the temperature and the Hessian region of the run that wrote it. The events found afterwards
get their $\nu_0$ with the current settings.

## Scratch files

- Each Hessian is written to a file whose name starts with `pykmc_htst_`, in `hessian_dir`.
  The file is deleted as soon as it is read, also when the calculation fails. Hessians are
  not kept.
- By default `hessian_dir` is the system temporary directory (it follows `$TMPDIR`). On a
  cluster, set `TMPDIR` to the local scratch of the job (for example `$SLURM_TMPDIR`) or give
  an absolute `hessian_dir`: the input file does not expand environment variables.
- A killed job can leave `pykmc_htst_*` files behind. They are safe to delete.
- Each session also writes a LAMMPS log `lammps.log.<n>-htst` (`n` is the session number) in
  the run directory. Every calculation rewrites it, so it holds the last one only. It is not
  written with `[Lammps] verbosity = 0` (which `[Control] verbosity = 0` implies when the
  `[Lammps]` key is not set).

## References

- G. H. Vineyard, *J. Phys. Chem. Solids* **3**, 121 (1957).
- T. Lazauskas, S. D. Kenny, R. Smith, *J. Phys.: Condens. Matter* **26**, 395007 (2014).
- Lefèvre López *et al.*, *Phys. Rev. Materials* **8**, 013609 (2024).
