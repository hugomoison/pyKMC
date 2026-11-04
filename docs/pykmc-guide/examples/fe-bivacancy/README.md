# Fe vacancy pairs

Goal: follow vacancy-pair motion in BCC Fe and distinguish basin acceleration from a failed attempt.

Prerequisite: [Installation](../../installation/README.md) and familiarity with a simpler vacancy example. Basin exploration can be expensive; Trillium avoids occupying your local computer.

Diagnostic status: the saved 20-step run completed, but all basin attempts fell back to ordinary events. Its software version also has a known energy-bookkeeping defect. Treat this as a diagnostic exercise; see [validation records](../../maintenance/validation.md#fe-and-nicr-diagnostic-status).

## 1. Understand the system

This configuration has 6746 Fe atoms and 6750 ideal BCC sites in a periodic 42.8295 Å cube: four vacancies, initially arranged as two pairs. Each initial pair has a separation of approximately `a = 2.8553 Å`, the BCC second-neighbor distance.

Preparation copies `input.in`, `initial_config.xyz` and `FeC-Becquart2013.eam` from the installed `examples/Fe_bcc_6746at_bivacancies`. The potential file includes Fe–C interactions, but this structure uses only Fe atoms.

<a id="3-understand-the-purpose-of-basin-acceleration"></a>
<a id="basin"></a>

### Basins and flickering

Flickering is repeated low-barrier motion between nearby minima without comparable long-range progress. Basin acceleration attempts to account for those internal transitions and select an exit with an associated residence time. It does not search the entire potential-energy landscape exhaustively, and `global` does not mean that every possible atomic configuration is enumerated.

In the recorded implementation, basin detection tests the selected forward barrier and an available reverse barrier against `energy_thr`. This threshold is a grouping criterion in eV, not the pARTn force tolerance in eV/Å, and not a cutoff that deletes all low-barrier events. See the [versioned detector](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/pykmc/basins/detection.py).

Relate each saved transition to row `s` of `pykmc.out`, candidates in step `s` of `pykmc.events`, and the search/refinement diagnostics in `pykmc.info`. A successful basin exit may represent several internal transitions; it need not be one nearest-neighbor hop. For the column definitions, use the [shared output explanation](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time).

If basin attempts keep failing, preserve the input/version/logs and report the stage of failure. A separate `basin = False` control can test ordinary KMC, but cannot validate the basin objective of this chapter. Recombination is not the objective: this starting structure has vacancies, not a vacancy–SIA pair.

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
cd ~/pykmc-guide/examples/fe-bivacancy &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/fe-bivacancy-01 &&
cd ~/pykmc-runs/fe-bivacancy-01 &&
ls -lh input.in initial_config.xyz FeC-Becquart2013.eam
```

The inputs are now together in `~/pykmc-runs/fe-bivacancy-01`. Open the settings:

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
cd ~/pykmc-guide/examples/fe-bivacancy &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/fe-bivacancy-01" &&
cd "$SCRATCH/pykmc-runs/fe-bivacancy-01" &&
ls -lh input.in initial_config.xyz FeC-Becquart2013.eam
```

This is the submission directory. Open the settings:

```bash
nano input.in
```

</details>

Preparation also saves `input.in.upstream` and `guide-preflight.py` automatically. It leaves numerical settings unchanged. Apply the input checklist below before launching. In nano, save with Ctrl+O, Enter, then exit with Ctrl+X.

<a id="1-prepare-and-review-the-inputs"></a>
<a id="input"></a>

## 3. Input checklist

Example parameters are under maintainer review. Required run settings are listed below; [validation](../../maintenance/validation.md#current-input-settings) records numerical settings and source versions. Preparation copies the installed input unchanged.

### Required files and run layout

- Set `initial_config = ./initial_config.xyz`.
- Use `pair_style = eam/alloy` and `pair_coeff = * * ./FeC-Becquart2013.eam Fe`.
- Set `n_steps = 20` for this diagnostic.
- Use `basin = True`, `active_volume = False` and `recycle = False`. The recorded diagnostic uses `[Basin] style = global` with `energy_thr = 0.6` eV.
- Set `[AtomicEnvironment] rnei = 3.2` Å. Preparation preserves the supplied 2.8 Å value; this is the one deliberate recognition-parameter change used by this chapter's input check.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

Why 3.2 Å? Ideal BCC Fe has 8 neighbors at about 2.473 Å and 6 more at 2.855 Å; 2.8 Å excludes the second shell, while 3.2 Å includes it and remains below the next shell at about 4.038 Å. This changes the recognition graph used by `cna/graph`; it does not change the EAM force cutoff. It is a system-specific diagnostic choice.

`rcut` controls the environment radius and `neighbors_add` extends graph-ID calculation to additional shells around noncrystalline atoms. The basin `energy_thr` helps identify low-barrier connections for basin treatment; it is separate from the force-convergence thresholds used in relaxation and saddle searches. The recorded basin behavior remains unresolved, as explained in the result checks.

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

If the input contains `[EventSearch] nnew_chance = 3`, it is ignored by the recorded version. `[pARTn] nnewchance` is a different, effective setting; moving the line there would change the calculation. Leave `[pARTn]` empty for this comparison. Check effective settings in `pykmc.log` and retain other recorded settings while interpreting this diagnostic.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/fe-bivacancy-01 &&
bash ~/pykmc-guide/examples/fe-bivacancy/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/fe-bivacancy/local/run-local.sh` to prevent idle sleep; leave the lid open.

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

Copy the [Slurm script](trillium/fe-bivacancy.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/fe-bivacancy-01" &&
cp ~/pykmc-guide/examples/fe-bivacancy/trillium/fe-bivacancy.sbatch job.sbatch
```

Open it and set `--account` to your allocation, `--mail-user` to your own email address, and `PYKMC_ROOT` to your cluster installation path. The supplied `--mail-type=BEGIN,END,FAIL` requests email notifications when the job starts, finishes or fails:

```bash
nano job.sbatch
bash -n job.sbatch
grep -nE 'YOUR_|/ABSOLUTE' job.sbatch
```

Both checks should print nothing; grep's exit status 1 means no placeholders matched. Keep the supplied process/session layout. The script's `12:00:00` limit is a maximum wall-time allowance, adjustable for your workload and current site policy. Resolve errors and placeholders before continuing.

With the same pyKMC environment active, check the prepared input on the login node before joining the queue. This reads the configuration and checks required files and the chapter's process layout; it does not start MPI or a simulation. Expect `PASS` before submission. The job repeats the check when it starts.

```bash
python -B guide-preflight.py --example fe-bivacancy &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/fe-bivacancy-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/fe-bivacancy-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/fe-bivacancy-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-check-completion-and-the-basin-outcome"></a>
<a id="results"></a>

## 5. Check completion and the basin outcome

Enter the result directory with pyKMC active:


```bash
python ~/pykmc-guide/examples/fe-bivacancy/scripts/check_fe_run.py trajkmc.xyz \
  --reference ~/pykmc-guide/examples/fe-bivacancy/Fe_bcc_perfect_6750_reference.xyz
```

The checker reads the adjacent `input.in`, `pykmc.out`, `pykmc.log` and, when present, `pykmc.events`. It checks frame/step counts, cell/composition, finite progress values, time increments, and four empty sites with zero excess atoms in each saved frame. Its structural success line is:

```text
PASS: saved structure and progress consistency
```

Read the following `REVIEW` lines too. A zero checker exit code does not certify basin acceleration or physical time. Require local launcher status `0`; on Trillium also require final Slurm `COMPLETED` / `0:0`. In particular:

- `Exploring the Basin` means an attempt started, not that a valid accelerated exit was applied.
- `Basin fails ... back to original event` means exploration failed and pyKMC used the ordinary selected event.
- `Reconstruction Exit State Basin fails ...` means an exit was proposed but its reconstruction failed; this also falls back.
- A stored `basin_connectivity_*.pickle` alone is not proof of an applied exit. Inspect the basin-exit record in `pykmc.events` and its surrounding log messages.

The saved cluster run used a version with a basin-fallback energy-bookkeeping defect. A large `dT`, disappearance of low-barrier candidates or many `INVALID_DE` rejections therefore requires review; do not interpret it immediately as slow physical diffusion. The [version notes](../../maintenance/validation.md#fe-and-nicr-diagnostic-status) identify the existing upstream fix without changing your installation.

<a id="4-visualize-vacancy-motion-in-ovito"></a>
<a id="visualization"></a>

## 6. Visualize vacancy motion in OVITO

Use the supplied [6750-site BCC Fe reference](Fe_bcc_perfect_6750_reference.xyz), aligned with this cell and ready to load.

1. Open `trajkmc.xyz`, add Wigner–Seitz defect analysis, choose External file, and load the Fe reference at constant frame 0. Keep per-type occupancy and affine mapping off for this fixed matching cell.
2. Choose Sites, and inspect the unfiltered counts first: the saved run retained four vacancies and zero interstitials.
3. Add Expression selection with `Occupancy != 0`, then Delete selected. The remaining four points are empty reference sites, not four moving Fe atoms.
4. Step through the trajectory and compare pair separations and motion; account for periodic crossings. Use a second unfiltered pipeline to inspect the real atoms. Atoms output cannot show an empty vacancy site.

The [OVITO WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html) explains the occupancy method. If the cell, orientation or upstream input changes, review the reference before interpreting extra defects.

## Completion criteria

You can distinguish vacancy motion, basin attempts, applied exits and fallback, and document unresolved warnings. Successful basin acceleration still needs a clean rerun and reviewed kinetics.
