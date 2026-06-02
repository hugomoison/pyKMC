# HTST Rate Prefactors (Vineyard ν₀)

The rate-constant prefactor is configured in the `[RateConstant]` section of the
input file. Two styles are available:

- **`constant`** (default behaviour): `k = k0 · exp(-ΔE / kᵦT)` with a fixed,
  user-set `k0`.
- **`htst`**: `k = ν₀ · exp(-ΔE / kᵦT)`, where `ν₀` is the **Vineyard harmonic
  prefactor** computed per reference event from the actual vibrational spectrum
  of the transition — the publication-standard Harmonic Transition State Theory
  prefactor, replacing the hand-set `k0`.

Example:

```INI
[RateConstant]
style = htst
k0 = 1e12          # per-event fallback when ν₀ cannot be computed
T = 300.0
free_radius = 6.0
fd_step = 0.01
nu0_min_THz = 1.0
nu0_max_THz = 100.0
```

`style = constant` is unchanged from previous versions and remains the default
recommendation when you do not need entropic (prefactor) corrections.

---

## The Vineyard prefactor

Vineyard (1957) showed that, under the harmonic approximation, the rate of
crossing a first-order saddle is

$$
k(T) = \frac{\prod_{i=1}^{3F} \nu_i^{\text{min}}}{\prod_{j=1}^{3F-1} \nu_j^{\text{sad}}}
\cdot \exp\!\left(-\frac{E_a}{k_B T}\right)
\;=\; \nu_0 \cdot \exp\!\left(-\frac{E_a}{k_B T}\right)
$$

where the $\nu^{\text{min}}$ are the positive normal-mode frequencies at the
initial-state minimum and the $\nu^{\text{sad}}$ are the positive modes at the
saddle (the single imaginary mode is dropped). The activation energy $E_a$ is
unchanged — it still comes from the pARTn saddle search. HTST only replaces the
prefactor.

ν₀ is computed **once per reference event** (forward *and* backward) and stored
in the `nu0` column of the reference table; reconstructed/refined events inherit
it, exactly as they inherit the barrier.

## Partial Hessians and the free region

For large systems the full $3N$ Hessian is impractical and is dominated by bulk
phonons that cancel in the ν₀ ratio. pyKMC therefore uses a **partial Hessian**:
the atoms within `free_radius` of the moving atom are free (vibrating), and the
rest are **frozen at their relaxed positions**. The Hessian is built by central
finite differences of the forces (`fd_step`) through the LAMMPS engine.

Freezing the boundary removes the translational zero modes, so the minimum's
partial Hessian is positive-definite and the saddle's has exactly one negative
mode — cleaner and more robust than a free cluster.

## Diagnostics and fallback

ν₀ never crashes a run. If an event's saddle Hessian does not have exactly one
negative mode, the minimum Hessian is not positive-definite, the engine errors,
or ν₀ falls outside `[nu0_min_THz, nu0_max_THz]`, that event **falls back to
`k0`** and the reason is logged to the HTST diagnostics log. Inspect that log to
audit how many events used ν₀ vs the `k0` fallback.

| Knob | Default | Meaning |
|---|---|---|
| `free_radius` | 6.0 Å | Radius of the free (vibrating) region |
| `fd_step` | 0.01 Å | Finite-difference displacement |
| `nu0_min_THz` / `nu0_max_THz` | 1 / 100 | Acceptance window; outside → `k0` fallback |
| `require_one_negative_mode` | True | Reserved; the saddle check is always enforced |

## Basin limitation (v1)

Basin super-events route through the same rate dispatcher but currently use the
`k0` fallback under `style = htst` (logged once per run). Per-state ν₀ inside a
basin is a documented future extension.

## Offline enricher

To compute ν₀ for an existing reference table without a live run (e.g. to
inspect/curate prefactors), use the offline enricher. It treats each stored
event's neighbour subset as a frozen-boundary cluster and runs on a serial
in-memory LAMMPS engine — no MPI required:

```bash
python -m pykmc.htst.enrich \
    --reference-table reference_table.pickle \
    --potential NiAlH_jea.eam \
    --out reference_table_htst.pickle \
    --report nu0_report.csv
```

v1 supports single-element systems. For small stored subsets, reduce
`--free-radius` so the free atoms keep complete neighbour shells inside the
subset.

## Validation

On the `NiAlH_jea.eam` potential, the engine reproduces the independent
analysis-side canonical values: **ν₀ ≈ 13 THz for the Ni(100) surface_1NN hop**
(`Ea ≈ 0.6 eV`), with all surface/subsurface events landing in the physical
FCC-Ni range of 5–30 THz. See `tests/htst/test_engine_prefactors.py` and the
cross-reference toolchain in `apps/PyKMC_Analysis/Analysis/HTST.md`.

## Known limitations

- **Surface soft modes inflate ν₀ (important).** On surfaces, a partial Hessian
  at the default `free_radius = 6 Å` picks up low-frequency surface "rattle"
  modes that inflate ν₀ well above the physical 5–30 THz range (often > 100 THz).
  At the default `nu0_max_THz = 100` such events **safely fall back to k0** (the
  inflated ν₀ is rejected and logged). This was confirmed on a live
  Ni(100)+vacancy run: with the default bound every surface event fell back,
  while a wide-open bound produced finite (but inflated) ν₀; a clean small free
  region (≈4 Å) on the same events recovers the correct ~13 THz. The analysis
  side solves this by **auto-freezing** atoms above the mover (`mover_z + a/4`)
  for surface/subsurface motifs — that logic is **not yet ported to pyKMC v1**.
  Until it is, HTST is effectively inert on surface events (they use the k0
  fallback); for surface systems, either tune `free_radius` down or treat ν₀
  with care. Porting auto-freeze is the planned v1.1 enhancement.
- **Basin super-events** use the k0 fallback under `htst` (logged once per run).
- **κ recrossing** (Sharia & Henkelman) is out of scope in v1.
- ν₀ is harmonic and therefore **temperature-independent** — computed once per
  reference event and reused at all T.

## References

- G. H. Vineyard, *J. Phys. Chem. Solids* **3**, 121 (1957) — the harmonic
  prefactor formula.
- The recrossing correction κ (Sharia & Henkelman 2016) is **out of scope in
  v1**; the finite-difference Hessian along the imaginary mode is the natural
  extension point.
