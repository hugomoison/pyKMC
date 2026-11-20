# Si monovacancy

Goal: run 100 KMC steps and follow a vacancy in a diamond lattice.

Prerequisite: [Installation](../../installation/README.md). The [Ni monovacancy](../ni-monovacancy/README.md) chapter introduces the common output files and visualization method.

## 1. Understand the system

This structure has 511 Si atoms in a periodic 21.72 Å cube, based on 512 diamond-lattice sites with one atom removed. The Stillinger–Weber potential includes angular interactions.

Preparation copies `input.in`, `initial_config.xyz` and `Si.sw` from the installed `examples/Si_vac`.

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
cd ~/pykmc-guide/examples/si-monovacancy &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/si-monovacancy-01 &&
cd ~/pykmc-runs/si-monovacancy-01 &&
ls -lh input.in initial_config.xyz Si.sw
```

The inputs are now together in `~/pykmc-runs/si-monovacancy-01`. Open the settings:

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
cd ~/pykmc-guide/examples/si-monovacancy &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/si-monovacancy-01" &&
cd "$SCRATCH/pykmc-runs/si-monovacancy-01" &&
ls -lh input.in initial_config.xyz Si.sw
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

1. Set `initial_config = ./initial_config.xyz` instead of an old name such as `tmp.xyz`.
2. Use `pair_style = sw` and `pair_coeff = * * ./Si.sw Si`.
3. Set `n_steps = 100` as the teaching length, with `basin = False`, `active_volume = False` and `recycle = False`.
4. Delete the old `path_artnso` entry from `[pARTn]`, if present; the installed interface supplies the library.
5. Delete `r_evalf_max` from `[EventSearch]`, if present; it is ignored by the recorded version. Leave `[pARTn]` empty. Replacing it with `r_nevalf_max = 1000` would change the refinement evaluation limit from that version's default of 300.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

`style = diamond/graph` identifies diamond-lattice environments. `rnei` is the neighbor cutoff and `rcut` is the environment radius, separate from the SW interaction cutoff. `min_style` and `minimize` control relaxation; PSR/IRA settings control matching.

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

The copied input contains the numerical values for this run. Check the effective settings at the start of `pykmc.log`; use the validation record if your installed example differs. The compatibility edits above apply to the recorded software version.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/si-monovacancy-01 &&
bash ~/pykmc-guide/examples/si-monovacancy/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/si-monovacancy/local/run-local.sh` to prevent idle sleep; leave the lid open.

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

Copy the [Slurm script](trillium/si-monovacancy.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/si-monovacancy-01" &&
cp ~/pykmc-guide/examples/si-monovacancy/trillium/si-monovacancy.sbatch job.sbatch
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
python -B guide-preflight.py --example si-monovacancy &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/si-monovacancy-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/si-monovacancy-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/si-monovacancy-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-check-completion"></a>
<a id="results"></a>

## 5. Check completion

With pyKMC active, enter the result directory and use these checks for either platform:

1. Confirm `run-exit-status.txt` contains `0`, `pykmc.log` ends with `End of simulation`, and `pykmc.out` reaches step 100. On Trillium also require final Slurm `COMPLETED` / `0:0`; inspect the parent Slurm logs instead of local `run.log`.
2. Run the result checker:

```bash
python ~/pykmc-guide/examples/si-monovacancy/scripts/check_si_run.py trajkmc.xyz --steps 100 \
  --progress pykmc.out --reference ~/pykmc-guide/examples/si-monovacancy/Si_diamond_perfect_512_reference.xyz
```

Expected structural checks:

```text
PASS: 101 frames; 511 Si atoms and a fixed periodic 21.72 A cell
PASS: one vacant reference site and zero excess atoms in every frame
PASS: progress steps 0..100 with positive rates and increasing KMC time
```

The checker tests output consistency, not catalog completeness or kinetic accuracy. See the [shared output-file overview](../ni-monovacancy/README.md#5-check-the-completed-run). Individual saddle searches may fail; check accepted events and any refinement/reconstruction failures before judging the run.

<a id="3-connect-the-output-to-vacancy-motion"></a>
<a id="interpret"></a>

## 6. Connect the output to vacancy motion

- In `pykmc.out`, `Ea` and `k_evt` describe the selected event; `k_tot` sums the available rates. `dT` and `T` are simulated seconds; rates are in ps⁻¹. The launcher's `real` is computer time. See the [column and rate explanation](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time).
- The same reference-event ID can be reused at different atoms or in different orientations. A repeated ID alone does not mean the vacancy is stuck.
- `pykmc.events` lists candidates; `Refined` indicates whether each was refined at that step. A few low-barrier channels can dominate many high-barrier candidates, so count alone is not probability.
- `n_steps` counts transitions, not equal-duration MD frames. Random event choices and waiting times need not match another run.

For minimization checks, inspect the stopping criterion and final forces in `lammps.log.*`: [energy convergence or iteration limits](https://docs.lammps.org/minimize.html) may stop relaxation before the requested force tolerance is reached.

<a id="4-visualize-the-vacancy-in-ovito"></a>
<a id="visualization"></a>

## 7. Visualize the vacancy in OVITO

Use the supplied [512-site Si reference](Si_diamond_perfect_512_reference.xyz): ideal diamond, `a = 5.43 Å`, cell 21.72 Å. No generation step is needed. Neither a Ni FCC reference nor the vacancy-containing first frame is suitable.

1. Open `trajkmc.xyz` as the main source; expect frames 0–100. Disable old displacement/selection/color modifiers if reusing an OVITO session.
2. Add Wigner–Seitz defect analysis. Choose External file, load the supplied Si reference, and use a constant reference frame 0.
3. Choose Sites output, with per-type occupancy off and affine mapping off for this matching fixed cell. Check 1 vacancy and 0 interstitials before filtering.
4. Add Expression selection with `Occupancy > 0`, then Delete selected. Only the empty-site marker remains.
5. Step through frames and save the OVITO state if useful. The marker's color is cosmetic; it represents a vacant site, not a moving Si atom.

See the [OVITO WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html). Periodic crossings appear on the opposite box face; back-jumps are allowed. Nearest-neighbor site spacing is `a√3/4 ≈ 2.35 Å`. To view the moving atoms instead of the vacant site, follow the [displacement instructions](../ni-monovacancy/README.md#7-optional-inspect-which-atom-moved).

If WS reports many defects, first check the reference, cell, alignment and output mode. The [maintenance generator](../../maintenance/reference-generation/README.md) is only for rebuilding assets.

## Completion criteria

The run finishes, checks pass, and you can distinguish a vacancy-site hop from the atom moving into it. Keep your inputs/results and cite the [software and potential](../../references.md).
