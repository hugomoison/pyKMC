# Cu vacancy + SIA

Goal: run 100 KMC steps, distinguish vacancy and self-interstitial motion, and relate the trajectory to event rates. Recombination is a possible outcome within this run length.

Prerequisites: [Installation](../../installation/README.md) and familiarity with the [Ni SIA rate explanation](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time).

## 1. Understand the system

The periodic 46.93 Å cube contains 8788 Cu atoms and 8788 ideal FCC sites, with one vacancy and one SIA. Equal atom/site counts can therefore still describe defects.

Preparation reads the installed `examples/Cu_fcc_sia` and copies `input.in`, `Cu01.eam.alloy` and `cu_fcc_defects.xyz`, renaming that structure to `initial_config.xyz`. This selects the vacancy–SIA structure used by this chapter, with the contents unchanged. See the [potential references](../../references.md).

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
cd ~/pykmc-guide/examples/cu-vacancy-sia &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/cu-vacancy-sia-01 &&
cd ~/pykmc-runs/cu-vacancy-sia-01 &&
ls -lh input.in initial_config.xyz Cu01.eam.alloy
```

The inputs are now together in `~/pykmc-runs/cu-vacancy-sia-01`. Open the settings:

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
cd ~/pykmc-guide/examples/cu-vacancy-sia &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/cu-vacancy-sia-01" &&
cd "$SCRATCH/pykmc-runs/cu-vacancy-sia-01" &&
ls -lh input.in initial_config.xyz Cu01.eam.alloy
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

1. Set `initial_config = ./initial_config.xyz` to match the prepared filename. Preparation renames the copied structure, leaving its contents unchanged.
2. Use `pair_style = eam/alloy` and `pair_coeff = * * ./Cu01.eam.alloy Cu`.
3. Set `n_steps = 100` as the teaching length, with `basin = False`, `active_volume = False` and `recycle = False`.
4. If `[EventSearch]` contains `r_nnewchance = 3`, remove that entry: it is ignored by the recorded version. Moving it into `[pARTn]` would change the effective refinement setting from that version's default of zero. Leave `[pARTn]` empty for this walkthrough.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

For `cna/graph`, `rnei` sets the neighbor cutoff and `rcut` the environment-recognition radius. The EAM potential has its own interaction cutoff. Relaxation settings control how event endpoints reach minima; PSR/IRA settings control their matching.

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

The copied input contains the numerical values for this run. Check effective settings at the start of `pykmc.log`; use the validation record if your installed example differs. The run teaches defect motion and competing rates; 100 steps is a sampling choice, not a prescribed recombination time.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/cu-vacancy-sia-01 &&
bash ~/pykmc-guide/examples/cu-vacancy-sia/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/cu-vacancy-sia/local/run-local.sh` to prevent idle sleep; leave the lid open.

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

Copy the [Slurm script](trillium/cu-vacancy-sia.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/cu-vacancy-sia-01" &&
cp ~/pykmc-guide/examples/cu-vacancy-sia/trillium/cu-vacancy-sia.sbatch job.sbatch
```

Open it and set `--account` to your allocation, `--mail-user` to your own email address, and `PYKMC_ROOT` to your cluster installation path. The supplied `--mail-type=BEGIN,END,FAIL` requests email notifications when the job starts, finishes or fails:

```bash
nano job.sbatch
bash -n job.sbatch
grep -nE 'YOUR_|/ABSOLUTE' job.sbatch
```

Both checks should print nothing; grep's exit status 1 means no placeholders matched. Keep the supplied process/session layout. The script's `00:30:00` limit is a maximum wall-time allowance, adjustable for your workload and current site policy. Resolve errors and placeholders before continuing.

With the same pyKMC environment active, check the prepared input on the login node before joining the queue. This reads the configuration and checks required files and the chapter's process layout; it does not start MPI or a simulation. Expect `PASS` before submission. The job repeats the check when it starts.

```bash
python -B guide-preflight.py --example cu-vacancy-sia &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/cu-vacancy-sia-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/cu-vacancy-sia-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/cu-vacancy-sia-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-check-the-completed-run"></a>
<a id="results"></a>

## 5. Check the completed run

Enter the result directory with pyKMC active. Use the same checker on Trillium or a downloaded copy:


```bash
python ~/pykmc-guide/examples/cu-vacancy-sia/scripts/check_cu_run.py trajkmc.xyz \
  --reference ~/pykmc-guide/examples/cu-vacancy-sia/Cu_fcc_perfect_8788_reference.xyz
```

The first success line is:

```text
PASS: structure, progress and selected-event rate checks
```

The checker reports either `step budget reached; recombination not observed`, or recombination observed. It checks fixed-cell geometry, WS counts, progress/time consistency and the selected-event rate law. An early exit needs a defect-free final frame and the corresponding all-crystalline log message. Confirm the scheduler's successful exit independently.

Individual pARTn searches may fail while other searches supply valid events. Inspect `pykmc.info` for accepted searches and refinement failures, and `pykmc.log` for reconstruction problems. A passing structural check does not prove a complete event catalog or converged recombination kinetics.

<a id="3-connect-motion-to-rates"></a>
<a id="interpret"></a>

## 6. Connect motion to rates

Compare frame `s-1` → `s` in OVITO with row `s` of `pykmc.out` and the selected candidate in step `s` of `pykmc.events`. Use the shared [output-file overview](../ni-monovacancy/README.md#5-check-the-completed-run) and [column/rate explanation](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time).

- `k_evt / k_tot` is the selected candidate's probability; sum rates when comparing a whole family of equivalent events.
- An SIA can make many hops while the vacancy remains at one site. Compare their candidate rates before concluding that the vacancy is stuck.
- A repeated reference ID can describe motion at different atoms and orientations. Check the structures; reference IDs are not fixed labels for “vacancy” or “SIA” across runs.
- No recombination within 100 steps is not by itself a failure. Extending a separate run is optional; it is not needed to complete this introductory exercise.

`dT` and `T` are simulated seconds, not computer runtime. For a suspect event, inspect its forward/backward barriers and actual endpoint motion. In `lammps.log.*`, check final forces as well as the [minimization stopping criterion](https://docs.lammps.org/minimize.html); energy convergence need not satisfy the requested force tolerance.

<a id="4-visualize-both-defects-in-ovito"></a>
<a id="visualization"></a>

## 7. Visualize both defects in OVITO

Use the supplied [8788-site Cu reference](Cu_fcc_perfect_8788_reference.xyz): ideal FCC, `a = 3.61 Å`, cell 46.93 Å. It matches this example's cell; do not substitute a Ni reference or the defective first frame. No generation step is needed.

1. Open `trajkmc.xyz`. Add Wigner–Seitz defect analysis → External file, load the Cu reference, constant reference frame 0. Leave per-type occupancy and affine mapping off for this matching fixed cell.
2. Choose Sites. Before filtering, inspect vacancy/interstitial counts. Add Color coding for `Occupancy`, with a fixed range 0–2.
3. Add Expression selection: `Occupancy == 1`, then Delete selected. The remaining sites mark the vacancy (`0`) and excess occupancy (`2`). Step through frames; periodic crossings appear at the opposite face.
4. To see the SIA atoms, switch WS to Atoms and change the selection to `Occupancy <= 1`, followed by deletion. The two retained atoms share one site: they represent one excess atom, not two SIAs. Atoms mode cannot display an empty vacancy site.

These modes follow the [OVITO WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html). If recombination occurs, both counts become zero and the filtered view becomes empty; verify the full structure and termination log. Many unexpected defects suggest a reference/cell/alignment problem to investigate first.

## Completion criteria

The run terminates successfully, checks pass, and you can connect candidate rates with observed defect motion. Recombination within 100 steps is optional. Preserve your private results.
