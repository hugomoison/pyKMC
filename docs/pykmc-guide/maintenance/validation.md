# Validation scope and source records

Document updated: 2026-10-21. The current status below separates completed scientific runs from tests of guide scripts. Historical check counts are in the update record at the end.

## Current status

| Tutorial | Apple Silicon macOS | Trillium | WSL2 / other local platforms |
| --- | --- | --- | --- |
| Ni monovacancy, 100 steps / 8 ranks | Run, checks and visualization tested | Run, checks and visualization tested | Not yet validated |
| Ni FCC SIA, 10 steps / 8 ranks | Run, checks and visualization tested | Route supplied; no end-to-end cluster run reviewed | Not yet validated |
| Ni vacancy + SIA / 8 ranks; suggested 300-step budget | Execution and structural recombination confirmed; saved recombining and non-recombining trajectories checked | Shorter-budget execution and structural checks tested; revised 300-step route not separately run | Not yet validated |
| Si monovacancy, 100 steps / 8 ranks | Run completed; saved trajectory and progress checked; supplied WS reference geometry verified | Not yet validated | Not yet validated |
| Cu vacancy + SIA, 100 steps / 8 ranks; AV/basin/recycling off | 100-step baseline reviewed from downloaded results; the separately reported 1,000-step local run is outside this record | Run completed; saved structure, progress, candidate tables and rates checked | Not yet validated |
| Fe vacancy pairs, 20 steps / 8 ranks; rnei = 3.2, basin on | Downloaded-run analysis; local completion not validated | Completed with four vacancies retained, but all six basin attempts fell back; old-version energy bookkeeping limits kinetic interpretation | Not yet validated |
| NiCr slab, full/grey, 10 steps each / 8 ranks; basin on | Both runs completed; composition/cell/progress checked. Full had basin-exit reconstruction failures; grey had basin matching failures | Not yet validated | Not yet validated |

The seven systems in the inspected installed upstream example directory have chapters. Their completion/validation levels differ; a successful exit and structural checks do not establish converged material kinetics. Later upstream additions are outside this coverage until reviewed.

All seven chapters supply local and Trillium instructions. Availability of instructions is separate from end-to-end testing on each platform. Scientific parameters remain under maintainer review; the numerical record below describes the current walkthroughs.

## Current input settings

This is a compact record of settings used by the existing chapters, not a new parameter calibration. Read the copied `input.in` and the effective configuration printed in `pykmc.log`. Preparation reads the installed examples, so a newer checkout can differ from these records. Review a difference before treating it as the same tested setup.

All chapter launchers use 8 MPI processes, 7 sessions, `group_size = -1`, and no recycling or imported restart/catalog state. The step limits are teaching choices. All recorded inputs use the constant-prefactor rate model at `k0 = 10` ps⁻¹ and `T = 300` K.

| Chapter | Walkthrough length and features | Recorded recognition / search settings |
| --- | --- | --- |
| Ni monovacancy | 100 steps; basin/AV off | `cna/graph`; `rnei = 3.0`, `rcut = 6.5` Å; `nsearch = 20`; barrier range 0–5 eV |
| Ni FCC SIA | 10 steps (upstream one-step input extended); basin/AV off | `cna/graph`; `rnei = 3.0`, `rcut = 6.0` Å; `nsearch = 10`; barrier range 0–5 eV |
| Ni vacancy + SIA | Suggested 300 steps; basin off, AV on | `cna/graph`; `rnei = 3.0`, `rcut = 6.5` Å; `nsearch = 20`; barrier range 0–5 eV; `ract = 16.0`, `rmov = 9.0` Å |
| Si monovacancy | 100 steps; basin/AV off | `diamond/graph`; `rnei = 3.0`, `rcut = 6.3` Å; `nsearch = 50`; barrier range 0–3.5 eV |
| Cu vacancy + SIA | 100 steps; basin/AV off | `cna/graph`; `rnei = 2.9`, `rcut = 6.5` Å; `nsearch = 10`; `emin_event = 0.0` eV |
| Fe vacancy pairs | 20 steps; global basin on, AV off; `energy_thr = 0.6` eV | `cna/graph`; `rnei = 3.2` Å (chapter adjustment from 2.8), `rcut = 6.0` Å; `neighbors_add = 1`; `nsearch = 50`; barrier range 0–5 eV |
| NiCr slab | 10 steps each for full/grey; basin on, AV off; `energy_thr = 0.59` eV | `cna/graph`; `rnei = 3.0`, `rcut = 8.5` Å; `nsearch = 20`; barrier range 0–5 eV; `zseed = 12345` |

Fe's recorded LAMMPS relaxation uses `cg` and `minimize = 1e-10 1e-12 1000 1000`; its PSR/IRA values include `matching_score_thr = 0.3` and `kmax_factor = 2.0`. NiCr uses `cg` and `minimize = 1e-10 1e-12 10000 10000`, with `matching_score_thr = 0.4` and `kmax_factor = 2.0`. These are historical diagnostic settings, not evidence that the basin limitations have been resolved. Preserve the full input and software records with each result.

When maintainers revise parameters, check the chapter instructions, prepared inputs, `examples/_shared/preflight.py` constraints and this record together. A passing preflight checks compatibility with the chapter's launch layout; it does not calibrate physical parameters.

## Fe and NiCr diagnostic status

**Fe:** the completed cluster run used clean pyKMC `e317491932f40eeacbf2829ea2a11c717126eab9`, 20 steps, eight ranks/seven sessions, default group size, and the original template with only `rnei` changed from 2.8 to 3.2 Å. All 21 saved frames contain 6746 Fe atoms, four vacant sites and zero excess atoms relative to the supplied 6750-site BCC reference. The initial vacancy pairs have second-neighbor separation in this checked file, despite the upstream README's 1NN description. Six basin attempts all fell back; no applied basin-exit record was found. One step has a very large physical-time increment after low-barrier channels disappear. Treat that timing as unresolved, not a measured diffusion time.

The older source does not update `total_energy` on two basin-fallback paths. Upstream commit [a4986d0](https://github.com/hugomoison/pyKMC/commit/a4986d04bd7f0a8d114332324998b88bfc744645) adds those assignments; the recorded local `551a822` includes it. This identifies a version-specific bookkeeping defect, not a complete explanation or fix for every basin exploration failure. Do not promote the old cluster run to kinetic validation or silently update an active installation. A reviewed rerun containing the fix is still needed; no such rerun was performed for this chapter.

**NiCr:** both local runs used recorded pyKMC `551a822acabc3572d5fbc8d57e2c7ce318fb5a16`, the installed template's ten-step/eight-rank recipe, and the same initial configuration/potential. Their effective inputs differ only in `atom_coloring_mode`. Each saved trajectory has 11 frames, 360 Ni + 87 Cr, a fixed slab cell and unchanged species ordering. Full recorded four applied basin exits followed by six exit-reconstruction fallbacks (`RECONSTRUCTION_INVALID_MIN2`); grey recorded ten basin matching fallbacks (`PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD`) and no applied exit. The upstream clean-full/failing-grey promise is therefore only partly reproduced. The log contrast supports a diagnostic comparison, not unique causal proof for every rejected event or quantitative alloy kinetics.

**New helper validation:** preparation and launcher/batch protections are tested with isolated fixtures, not fresh scientific simulations. Checkers are run on the saved Fe/full/grey data and negative fixtures. Fe reference geometry is checked against the initial configuration and all saved frames. NiCr visualization follows OVITO's species/displacement workflow; no WS count, fresh GUI walkthrough or new cluster execution is claimed. The shared checker reads text/XYZ files and never loads pickle files. Keep the entire guide together because the two checker entry points use `examples/_shared/`.

The Fe batch template uses the standard module entry point; the historical cluster run used an exception-reporting diagnostic wrapper. The new NiCr launcher also adds input/version checks. Their fixture tests do not establish end-to-end execution on every platform. Do not change search, force or basin thresholds merely to force a passing chapter.

The pair tutorial's 300-step budget is a teaching choice, not a separate freshly measured run or a guaranteed encounter time. Its early-stop checker and preparation/launcher entry points were tested with saved results and isolated fixtures; no new scientific run was launched for the chapter revision. Recombination was observed in a longer-budget local run that ended early. The temporary loss of low-barrier channels and associated large KMC time increments remain under investigation; structural success does not validate lifetimes.

## Recorded working local source configuration

The local tutorials used Python 3.13 on Apple Silicon and pyKMC's `develop` branch at the following source revision. The installer is pinned separately at `599f162e63ae8bbecf84c48d0a22b8053dae4fe7` in the Kerrator fork.

| Component | Source commit / local change |
| --- | --- |
| pyKMC | `551a822acabc3572d5fbc8d57e2c7ce318fb5a16` |
| LAMMPS | `9f06a79b033159c606ca880ee63f0ec82b37dc3a` |
| IRA | `3cb0c299e2e664f8131948b90f9926869d42459c` |
| pARTn | `edea36aca8215a1d484b3b8695ecb9676fe56498` plus the [conditional interface-export patch](../installation/patches/partn-public-c-bindings-edea36a.patch) |

These are source records, not a fully locked build: compilers and Python dependencies can also change. Do not switch source revisions underneath an existing editable installation and assume its compiled libraries changed too. Reproducing an older setup requires a separate reviewed build and fresh checks. The [recovery appendix](../installation/macos-partn-recovery.md) explains the patch's narrow applicability.

The Trillium route uses the module stack listed in the installation chapter. A complete dependency-and-patch manifest for that route is still needed before advertising an exactly reproducible pinned cluster build. No local source record above is implicitly assigned to Trillium.

The checked Cu cluster run recorded pyKMC `e317491932f40eeacbf2829ea2a11c717126eab9` with a clean tracked/untracked status report. This does not supply the missing full dependency/build manifest. Its source example was `examples/Cu_fcc_sia`; the tutorial copies the structure as `initial_config.xyz` and removes an ignored `[EventSearch] r_nnewchance` entry without changing its effective pARTn setting.

## Installer source selection

Source review on **2026-10-21** compared the pinned Kerrator installer `599f162` with upstream `develop` at `551a822` and `main` at `9fd1f6e`. Both upstream Mac scripts still execute a Python 3.13-specific replacement/check for `<3.13,` / `<3.14,`, while the fetched pyKMC metadata declares `requires-python = ">=3.10"`. The check therefore fails for that combination. This is a source-level finding, not a fresh installation test. See the [upstream Mac check](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/docs/install/install_pykmc_mac.sh#L109-L120) and [project metadata](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/pyproject.toml#L9).

The selected fork scripts remove that obsolete step, pin pARTn/IRA, record resolved source revisions and use `pykmc_install/` rather than the upstream scripts' `pykmc/`. The Mac script additionally supports the guide's `MAKE_JOBS` setting and retains build-error logs. Both fork scripts still clone **pyKMC itself from the upstream hugomoison/pyKMC repository**. Their existence does not establish that every platform has passed a fresh install; retain the separate validation scope above.

The long-term preference is an upstream-maintained installer containing the needed fixes, at a reviewed revision. Switch after checking installation, activation paths and a small example on the affected platforms; never substitute URLs alone or silently upgrade a working installation. This revision changes neither installer downloads nor software versions.

## Current guide checks

Latest revision: submission-time input checks, non-conflicting installation anchors, clearer input sections and publication naming. Mandatory command blocks remain collapsible at the reader's request.

The latest regression run passed 1,339 assertions, including invalid-input rejection before the scheduler stub and the NiCr grey submission. Static checks passed for 166 internal links/anchors and 152 Bash command blocks (Bash and zsh). The 31 existing foldouts were retained. Fifteen ignore-rule cases keep generated run files out while preserving reference XYZ files, scripts and illustrations.

A local browser preview confirmed that the old Trillium activation anchor and the Windows activation heading lead to distinct sections, and that the expanded Trillium block renders the pre-submit command. This was a local GFM-compatible preview, not a GitHub-hosted publication. No simulation, installation change or real Slurm submission was performed.

Run the regression checks from [maintenance/tests](tests/README.md) after changing scripts or command examples. The suite uses temporary files and replaces simulation/engine/scheduler launches with test fixtures. Scientific runs and platform installations must be validated separately.

## Guide layout and maintenance

Each chapter contains preparation, input checks, launch commands, result interpretation and visualization. Keep the complete guide together: shared scripts and reference assets are dependencies. Short route-directory README files point to the chapter; executable paths remain unchanged.

The portable top-level folder name is `pykmc-guide`. Example directories use lowercase hyphen-separated names; Python modules and preparation/checker scripts retain their existing underscore names, and local launch scripts use `run-local.sh`. Preserve upstream potential/configuration names and reference-asset names where required by commands. See the [publication checklist](publishing.md) before committing.

If an older prepared directory lacks `guide-preflight.py`, prepare a new destination and reapply the chapter's input checklist. Keep earlier results unchanged. Guide updates do not update the installed pyKMC code or dependencies.

## Update record

### 2026-10-21 — publication preparation

Kept the local/Trillium foldouts. Added input preflight before `sbatch` (including NiCr grey), clarified email notifications, separated run requirements from physical-parameter explanations, added the first chapter's progress-column explanation, resolved the Windows/Trillium anchor collision, and documented the guide folder location. No physical input, installation or simulation was changed.

### 2026-10-21 — earlier layout and helper revisions

The earlier helper correction passed 1,263 assertions. The subsequent language/layout snapshot passed 1,260 assertions; it also checked 161 internal links, 150 Bash command blocks and 16 isolated preparation commands. These are counts from different snapshots, not the latest test total. The local browser preview used GFM-compatible rendering rather than GitHub-hosted pages.

Preparation began copying the shared input checker for both launch routes. The installed local `551a822` and historical cluster `e317491` have identical `config.py` contents; compatibility with future revisions still needs checking. Fe/NiCr checks began reporting unknown basin wording and checking NiCr type/mass/EAM assignments. Rechecking the saved runs preserved their existing diagnostic warnings.

Fe/NiCr chapters and complete execution instructions for all seven examples were added. The new launch scripts were tested with isolated fixtures, not new scientific calculations. Local timer fallback, exit-status propagation, overwrite protection and shared input checks were included.

### 2026-10-19 — Cu chapter

The saved cluster run, candidate rates and 8788-site reference were checked. The reviewed run used a diagnostic Python entry point; the supplied standard-entry-point batch script was tested with fixtures rather than an end-to-end cluster run. No local 1,000-step result is included in this validation record.

### 2026-10-16 — Si chapter

The saved 100-step trajectory, progress output and 512-site reference passed structural checks. Preparation and launcher behavior were tested with fixtures; local LAMMPS `sw` availability was checked separately.

### 2026-10-07 — earlier Ni checks

Saved Ni results and wrapper protections were checked. Local library loading, `eam/alloy`, and the two-process MPI communication test passed. The pair chapter's 300-step suggestion is a teaching limit; observed recombination and unresolved timing limitations are described above.
