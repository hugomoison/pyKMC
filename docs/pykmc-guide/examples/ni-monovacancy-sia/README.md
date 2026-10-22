# Ni vacancy + SIA

Goal: follow a vacancy and a self-interstitial atom (SIA), and recognize recombination from structure, energy and logs.

Prerequisites: [Ni monovacancy](../ni-monovacancy/README.md) and [Ni FCC SIA](../ni-fcc-sia/README.md).

Use a 300-step limit. Recombination may occur earlier or remain unobserved; this chapter teaches structural identification. Temporary loss of low-barrier channels in saved runs limits lifetime interpretation, as explained below.

## 1. Understand the system

The periodic 35.2 Å cube has 4000 Ni atoms and 4000 ideal FCC sites**, but initially one site is empty and another contains two atoms: one vacancy plus one excess atom, a Frenkel pair. Equal atom/site counts do not mean a defect-free structure. During recombination the defects disappear; the atom count remains 4000.

Preparation copies `input.in`, `initial_config.xyz` and `Ni_v6_2.0_LKBeland2016.eam` from the installed `examples/Ni_fcc_4000at_monovacancy+sia`.

<a id="prepare"></a>

## 2. Prepare a run

Choose one platform below. Commands assume the guide is at `~/pykmc-guide`; substitute your actual guide and installation paths if different. You can use Tab completion while entering paths ([path help](../../paths.md)). Keep the complete guide in place and let preparation create the new destination.

<a id="local"></a>

<details>
<summary>Local computer — activate and prepare</summary>

Use your local terminal; on Windows, use Ubuntu after the [WSL setup](../../installation/README.md#b-windows--wsl2--ubuntu).

On macOS, activate:

```bash
source ~/software/pyKMC_setup/pykmc_install/pykmc_env/bin/activate &&
unset PYTHONPATH
```

On Linux/WSL, activate your Linux installation instead:

```bash
source ~/software/pyKMC_setup/pykmc_install/activate.sh
```

Check that Python and MPI belong to the intended installation/toolchain:

```bash
command -v python mpirun
```

Python should be in this installation's `pykmc_env/bin/`. Resolve activation errors before preparing:

```bash
cd ~/pykmc-guide/examples/ni-monovacancy-sia &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/ni-monovacancy-sia-01 &&
cd ~/pykmc-runs/ni-monovacancy-sia-01 &&
ls -lh input.in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam
```

The inputs are now together in `~/pykmc-runs/ni-monovacancy-sia-01`. Open the settings:

```bash
nano input.in
```

</details>

<a id="trillium"></a>

<details>
<summary>Trillium — activate and prepare</summary>

Use the installation and complete guide from [Trillium setup](../../installation/README.md#c-trillium). These commands run in your SSH terminal:

```bash
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0 &&
source "$SCRATCH/pyKMC_install/pykmc_install/activate.sh" &&
command -v python
```

Keep the module-provided `PYTHONPATH`. After successful activation:

```bash
cd ~/pykmc-guide/examples/ni-monovacancy-sia &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/ni-monovacancy-sia-01" &&
cd "$SCRATCH/pykmc-runs/ni-monovacancy-sia-01" &&
ls -lh input.in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam
```

This is the submission directory. Open the settings:

```bash
nano input.in
```

</details>

Preparation also saves `input.in.upstream` and `guide-preflight.py` automatically. It leaves numerical settings unchanged. Apply the input checklist below before launching. In nano, save with Ctrl+O, Enter, then exit with Ctrl+X.

<a id="1-prepare-and-understand-the-system"></a>
<a id="input"></a>

## 3. Input checklist

Example parameters are under maintainer review. Required run settings are listed below; [validation](../../maintenance/validation.md#current-input-settings) records numerical settings and source versions. Preparation copies the installed input unchanged.

### Required files and run layout

- Set `initial_config = ./initial_config.xyz`.
- Use `pair_style = eam/alloy` and `pair_coeff = * * ./Ni_v6_2.0_LKBeland2016.eam Ni`.
- Set `n_steps = 300` as the teaching length. Recombination may occur earlier or require a longer trajectory.
- Use `basin = False`, `active_volume = True` and `recycle = False`; keep the corresponding `[ActiveVolume]` section.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

An [active volume](../../glossary.md#active-volume) limits local calculations to a spherical region. `rmov` is the movable-region radius; `ract` is the entire-region radius. The shell between them is frozen. The recorded setup uses a 7 Å buffer; the actual radii are listed in validation. These radii serve a different purpose from `rnei` (neighbor cutoff) and `rcut` (environment-recognition radius).

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

Structural relaxation and PSR/IRA matching settings also affect whether an event can be reconstructed in the current system. This chapter uses the recorded input and retains its [physical-time limitations](#scope-of-the-physical-time-interpretation); increasing the step limit alone does not establish reliable recombination kinetics.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/ni-monovacancy-sia-01 &&
bash ~/pykmc-guide/examples/ni-monovacancy-sia/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/ni-monovacancy-sia/local/run-local.sh` to prevent idle sleep; leave the lid open.

When the command returns:

```bash
cat run-exit-status.txt
tail -n 20 pykmc.log
tail -n 5 pykmc.out
```

Expect status `0`. Launch output is in `run.log`; software records are in `run-versions/`. The printed `real` is elapsed computer time. Continue with the result checks below from this directory.

</details>

<details>
<summary>Trillium — submit, monitor and download</summary>

Copy the [Slurm script](trillium/ni-monovacancy-sia.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/ni-monovacancy-sia-01" &&
cp ~/pykmc-guide/examples/ni-monovacancy-sia/trillium/ni-monovacancy-sia.sbatch job.sbatch
```

Open it and set `--account` to your allocation, `--mail-user` to your own email address, and `PYKMC_ROOT` to your cluster installation path. The supplied `--mail-type=BEGIN,END,FAIL` requests email notifications when the job starts, finishes or fails:

```bash
nano job.sbatch
bash -n job.sbatch
grep -nE 'YOUR_|/ABSOLUTE' job.sbatch
```

Both checks should print nothing; grep's exit status 1 means no placeholders matched. Keep the supplied process/session layout. The script's `01:00:00` limit is a maximum wall-time allowance, adjustable for your workload and current site policy. Resolve errors and placeholders before continuing.

With the same pyKMC environment active, check the prepared input on the login node before joining the queue. This reads the configuration and checks required files and the chapter's process layout; it does not start MPI or a simulation. Expect `PASS` before submission. The job repeats the check when it starts.

```bash
python -B guide-preflight.py --example ni-monovacancy-sia &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/ni-monovacancy-sia-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/ni-monovacancy-sia-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/ni-monovacancy-sia-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-check-how-the-run-ended"></a>
<a id="results"></a>

## 5. Check how the run ended

In the result directory, with pyKMC active:

1. Local: check `run-exit-status.txt` is `0`, and inspect `run.log`. Trillium: check `sacct` reports `COMPLETED` / `0:0` and inspect the parent submission directory's `slurm-JOBID.out/.err`; analysis files are inside `run-JOBID/`. The batch script also records `run-exit-status.txt` inside `run-JOBID/`; its console output is in the parent Slurm logs, not `run.log`.
2. For either route, inspect the end of `pykmc.log`, the last rows of `pykmc.out`, and run the [read-only checker](scripts/check_pair_run.py):

```bash
python ~/pykmc-guide/examples/ni-monovacancy-sia/scripts/check_pair_run.py trajkmc.xyz \
  --input input.in --progress pykmc.out --log pykmc.log \
  --reference ~/pykmc-guide/examples/ni-monovacancy-sia/Ni_fcc_perfect_4000_reference.xyz
```

The checker reads the step limit from `input.in`; it does not require 301 frames if recombination ended the run early. It checks atom/cell consistency, periodic WS occupancy, output rows and Arrhenius rates. It prints one of these outcomes:

```text
PASS: structure, progress and selected-event rate checks
OUTCOME: recombination observed; inspect the reported transition in OVITO
```

or:

```text
PASS: structure, progress and selected-event rate checks
OUTCOME: step budget reached; recombination not observed
```

### Recombination and early stopping

In the pyKMC version listed in the [validation records](../../maintenance/validation.md), the program saves the event's final frame and exits when all atomic environments are classified as crystalline. The log then contains:

```text
:=> Only atoms with cristalline environment
:=> End of simulation
```

Confirm the accompanying 1 vacancy + 1 interstitial → 0 + 0 WS change, unchanged atom count, and the energy change. `End of simulation` or exit code 0 alone is insufficient. The all-crystalline exit occurs before the normal end-of-loop restart save in this version, so a missing `restart_*.npz` alone does not imply failure.

### No recombination by the step limit

If all 300 steps finish and the pair remains, retain the result: this is a valid observation, not a failed installation. For another observation attempt, prepare a new run directory and choose a larger step limit, for example 1000, keeping other settings unchanged. That starts a new random trajectory; it does not continue the old one. Checkpoint continuation is outside this walkthrough. Do not change temperature or AV settings just to force an animation.

An unexplained early stop with defects still present needs investigation. Individual ARTn search failures can coexist with a successful run; read the search and refinement summaries in `pykmc.info`, not just failure banners.

<a id="3-connect-the-files-to-the-physics"></a>
<a id="interpret"></a>

## 6. Connect the files to the physics

Reuse the [output column definitions and rate equations](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time). Row `i` of `pykmc.out` describes frame i−1 → frame i.

| File | What to inspect here |
| --- | --- |
| `input.in` and the beginning of the launch log | Temperature, prefactor, AV settings, step limit and effective defaults. |
| `pykmc.out` | Executed event ID, forward barrier, selected/total rates, physical waiting time and final potential energy. |
| `pykmc.events` | Candidates for that step. `Selected` is the row index in that step's table; `Ref Event` identifies a reusable reference event. These are different numbers. Compare barriers, rates and `Refined`. |
| `pykmc.info` | New environments, attempted/successful searches, accepted reference events and refinements. More searches can be needed as the defects approach. |
| `trajkmc.xyz` | Event endpoints: establish whether an event moves the SIA, moves the vacancy, or annihilates both. IDs are not universal mechanism labels. |

If recombination occurs at frame `j`, compare `j−1` and `j` and the corresponding output row. A decrease in potential energy supports defect removal; it is not the activation barrier. Several atoms can participate in the final rearrangement, and endpoint snapshots do not resolve the saddle-crossing path.

The defects need a suitable encounter configuration, not a prescribed number of jumps. Backtracking, separation and orientation matter; they are not moving deliberately toward each other. See the [recombination background references](../../references.md#frenkel-pair-recombination).

### Scope of the physical-time interpretation

Recombination can be established structurally without establishing a reliable recombination lifetime. In the reviewed Ni vacancy + SIA runs, some steps temporarily lost low-barrier SIA channels from the active catalog and retained only slow channels, producing a large time increment. The cause remains under investigation; this is not proof of either an AV error or a complete catalog.

If the checker warns about a time jump, inspect `pykmc.events` and `pykmc.info` before, at and after that step. A warning is a screening heuristic, not a diagnosis. Conversely, no warning is not proof of catalog completeness. Rounded `T(s)` values can repeat when tiny increments follow a huge one; inspect `dT(s)` rather than treating repeated printed totals alone as a frozen clock. Do not delete a large increment and report the remainder as a corrected physical lifetime. Keep inputs and source records when asking the [maintainers for help](../../README.md#support-and-references).

<a id="4-visualize-both-defects-and-their-disappearance"></a>
<a id="visualization"></a>

## 7. Visualize both defects and their disappearance

Open the run's `trajkmc.xyz`. Use the supplied [4000-site perfect reference](Ni_fcc_perfect_4000_reference.xyz) directly: it matches this aligned 35.2 Å cell. Do not generate another reference or replace the defective initial structure with it.

1. Disable old displacement modifiers. Add Wigner–Seitz defect analysis, choose External file, and load the reference. Use Sites, a constant reference, affine mapping off, and per-type occupancies off.
2. Add Color coding for `Occupancy`, fixed range 0 to 2.
3. Add Expression selection: `Occupancy == 1`, then Delete selected.

Initially two site markers remain: occupancy 0 for the vacancy and occupancy 2 for the SIA site. The colors depend on your palette. After confirmed recombination, no markers remain because every site has occupancy 1; disable deletion to reveal the intact crystal.

Atoms mode retains actual atoms, not empty sites. With the same filter, it shows the two occupants of the overoccupied site but cannot show the vacancy. Use Sites to track the vacancy, and Atoms to inspect the dumbbell. See the [OVITO WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html).

For the recombination motion itself, disable WS and its downstream selection/deletion/color modifiers. Use the original trajectory's Displacement vectors, referenced to the preceding frame, with minimum-image handling, then color by displacement magnitude. Check atom correspondence before interpreting vectors. This reveals moving atoms and nearby relaxation; it is not a vacancy map. Opposite box faces are connected periodically.

<a id="5-finish-the-exercise"></a>

## Completion criteria

Keep the inputs, logs, software records and OVITO state in your private records. You can explain the two defect markers, match a structural change to an event row, and classify your result as recombined, step limit reached without recombination, or needs investigation.
