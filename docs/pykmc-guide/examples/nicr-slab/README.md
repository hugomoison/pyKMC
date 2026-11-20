# NiCr surface slab

Goal: compare species-aware (`full`) and species-blind (`grey`) event matching for a surface vacancy in a Ni–Cr alloy.

Prerequisites: [Installation](../../installation/README.md) and the [basin explanation](../fe-bivacancy/README.md#basins-and-flickering).

Diagnostic status: both saved 10-step runs completed, but both had basin failures at different stages. The comparison illustrates recognition behavior; it has not established error-free basin exits. See [validation records](../../maintenance/validation.md#fe-and-nicr-diagnostic-status).

## 1. Understand the system

A slab is a thin crystal surrounded by vacuum along its surface normal. Here an FCC(100) slab has 447 atoms (360 Ni and 87 Cr), seven atomic layers and one missing top-surface site. The defect is still a vacancy; the new ingredients are the surface and chemical disorder.

The configuration has random Cr plus a deliberately patterned vacancy neighborhood, giving about 19.5% Cr. It is a teaching structure. Its cell is approximately `19.934754 × 19.934754 × 50.572 Å`, periodic in all three directions, with vacuum separating slab images.

Preparation copies `input.in`, `initial_config.xyz` and `Bonny_2013_NiFeCr.eam` from the installed `examples/NiCr_fcc_447at_slab_monovacancy`. Preserve the cell and boundary setup.

Run `full` first, then `grey` in a separate directory, changing only `atom_coloring_mode`. This changes graph hashing, matching and symmetry recognition, while retaining the species-dependent forces.

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
cd ~/pykmc-guide/examples/nicr-slab &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/nicr-slab-full-01 &&
cd ~/pykmc-runs/nicr-slab-full-01 &&
ls -lh input.in initial_config.xyz Bonny_2013_NiFeCr.eam
```

The inputs are now together in `~/pykmc-runs/nicr-slab-full-01`. Open the settings:

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
cd ~/pykmc-guide/examples/nicr-slab &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/nicr-slab-full-01" &&
cd "$SCRATCH/pykmc-runs/nicr-slab-full-01" &&
ls -lh input.in initial_config.xyz Bonny_2013_NiFeCr.eam
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
- Use `pair_style = eam/alloy` and `pair_coeff = * * ./Bonny_2013_NiFeCr.eam Cr Ni` with the recorded engine's type ordering.
- Set `n_steps = 10` for each comparison run.
- Use `basin = True`, `active_volume = False` and `recycle = False`.
- Use `atom_coloring_mode = full` first, then `grey` in a separate directory. Keep all other physical settings, including `zseed`, identical between the two runs.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

`atom_coloring_mode` controls whether graph hashing, matching and symmetry recognition distinguish chemical species. Switching to `grey` leaves Cr/Ni species-dependent forces in place. For `cna/graph`, `rnei` sets the neighbor cutoff and `rcut` the environment radius. The basin `energy_thr` controls the low-barrier grouping used by basin treatment; it is separate from force-convergence tolerances. `zseed` is a pARTn search seed.

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

The engine version in the validation records assigns element types alphabetically: Cr then Ni. The coefficient mapping must follow the engine's actual type order. The result checker verifies `labelmap`, `mass` and `pair_coeff` together in `lammps.log.0`, so preserve this log when downloading results. A missing or unsupported mapping stops verification.

Use the linked numerical record for this paired comparison; see the [versioned parameter documentation](https://github.com/hugomoison/pyKMC/blob/551a822acabc3572d5fbc8d57e2c7ce318fb5a16/docs/parameters_details.md) for further definitions.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/nicr-slab-full-01 &&
bash ~/pykmc-guide/examples/nicr-slab/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/nicr-slab/local/run-local.sh` to prevent idle sleep; leave the lid open.

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

Copy the [Slurm script](trillium/nicr-slab.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/nicr-slab-full-01" &&
cp ~/pykmc-guide/examples/nicr-slab/trillium/nicr-slab.sbatch job.sbatch
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
python -B guide-preflight.py --example nicr-slab &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/nicr-slab-full-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/nicr-slab-full-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/nicr-slab-full-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-run-a-controlled-fullgrey-comparison"></a>
<a id="grey"></a>

## 5. Repeat with grey

After the full run finishes, keep its results and prepare the second run using the same installation. Run the local jobs sequentially. A fixed pARTn seed helps control searches, but trajectories can still differ.

<details>
<summary>Local computer — prepare grey</summary>

```bash
cd ~/pykmc-guide/examples/nicr-slab &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/nicr-slab-grey-01 &&
cd ~/pykmc-runs/nicr-slab-grey-01 &&
nano input.in
```

Apply the same input checklist, changing only `atom_coloring_mode = grey`. Compare the files:

```bash
diff -u ../nicr-slab-full-01/input.in input.in
cmp ../nicr-slab-full-01/initial_config.xyz initial_config.xyz
cmp ../nicr-slab-full-01/Bonny_2013_NiFeCr.eam Bonny_2013_NiFeCr.eam
```

The diff should show only the coloring change (exit status 1 for that difference); both `cmp` commands should be silent and return 0. Resolve other differences, then launch:

```bash
cd ~/pykmc-runs/nicr-slab-grey-01 &&
bash ~/pykmc-guide/examples/nicr-slab/local/run-local.sh
```

</details>

<details>
<summary>Trillium — prepare and submit grey</summary>

With the same modules and environment active:

```bash
cd ~/pykmc-guide/examples/nicr-slab &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/nicr-slab-grey-01" &&
cd "$SCRATCH/pykmc-runs/nicr-slab-grey-01" &&
nano input.in
```

Apply the same input checklist, changing only `atom_coloring_mode = grey`. Compare:

```bash
diff -u ../nicr-slab-full-01/input.in input.in
cmp ../nicr-slab-full-01/initial_config.xyz initial_config.xyz
cmp ../nicr-slab-full-01/Bonny_2013_NiFeCr.eam Bonny_2013_NiFeCr.eam
```

Expect only the coloring difference and two silent `cmp` successes. After resolving differences, copy the already edited Slurm script from the full submission directory:

```bash
cp ../nicr-slab-full-01/job.sbatch job.sbatch &&
bash -n job.sbatch &&
python -B guide-preflight.py --example nicr-slab &&
sbatch job.sbatch
```

Monitor the new job ID and check/download its results using the commands above, replacing `nicr-slab-full-01` with `nicr-slab-grey-01`. Use a separate local download destination too.

</details>

Check each run below. Preserve both sets of inputs, structures, potentials and logs; event tables from the full run stay with that run.

<a id="3-check-results-and-interpret-the-warnings"></a>
<a id="results"></a>

## 6. Check results and interpret the warnings

In each completed result directory, with pyKMC active:

```bash
cat run-exit-status.txt
python ~/pykmc-guide/examples/nicr-slab/scripts/check_nicr_run.py trajkmc.xyz
```

Require launcher status `0`; on Trillium also check final Slurm `COMPLETED` / `0:0` and the parent Slurm logs. The checker reports frame/step consistency, the fixed slab cell and conserved species counts, then the warning categories. It does not compute a surface-vacancy diffusion coefficient or certify the event catalog. A structural `PASS` can legitimately be followed by `REVIEW` warnings.

Search for both matching and reconstruction failures:

```bash
grep -nE 'Basin fails|Reconstruction.*fails|PSR_MATCHING|INVALID_MIN2|back to original event' pykmc.log
```

| Observation | Interpretation |
| --- | --- |
| `PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD` | A proposed geometric match exceeds the acceptance threshold. It is rejected; the log alone does not identify a specific wrong-species atom. |
| `RECONSTRUCTION_INVALID_MIN2` / basin-exit reconstruction failure | The reconstructed result does not match the expected final minimum. This is distinct from a failed initial event search. |
| `back to original event` | The accelerated attempt was abandoned and the previously selected ordinary event applied. Normal program termination is still possible. |
| A repeated reference ID | Could represent reused events or back-and-forth motion; inspect endpoints and rates before calling it physical flickering. |

In the saved runs, `full` applied some basin exits and later fell back after exit-reconstruction failures; `grey` fell back after matching failures. Your counts may differ. Keep the input settings fixed while comparing modes; changing matching tolerance or disabling basin would change the question.

Use `pykmc.events` for candidate/reference information and any applied basin-exit records; use `pykmc.info` for search and refinement diagnostics. The [shared output guide](../ni-fcc-sia/README.md#6-read-barriers-rates-and-physical-time) explains barriers and rates. When a basin exit is applied, a saved step can represent more than one internal hop; do not assume a simple ordinary-event probability interpretation for every basin summary row.

<a id="4-visualize-the-slab-in-ovito"></a>
<a id="visualization"></a>

## 7. Visualize the slab in OVITO

Use actual atoms and species colors for this slab. A bulk perfect-crystal reference would place fictitious lattice sites in the vacuum.

1. Open `initial_config.xyz` or frame 0 of `trajkmc.xyz`. Use Particle types to give Ni and Cr contrasting colors; inspect top and side views. The side view shows the layers and vacuum; the top view shows the vacancy neighborhood.
2. Open each trajectory separately and retain species coloring. Step through frames and compare the local neighborhood and return motions. `full`/`grey` are pyKMC recognition settings, unrelated to OVITO's display colors.
3. Optionally add Displacement vectors with a constant upstream reference frame 0, apply minimum-image handling for periodic crossings, and inspect `Displacement Magnitude`. This shows net atom displacement between saved configurations, not a continuous transition path. It may also include relaxation from frame 0.
4. Correlate frame `s-1` → `s` with the output/log for step `s`. Separate runs will not necessarily select corresponding events at the same step number.

See [OVITO displacement vectors](https://www.ovito.org/manual/reference/pipelines/modifiers/displacement_vectors.html). Without particle IDs, displacement analysis relies on consistent atom ordering; the checked writer preserves system order, but review this if changing output formats. Do not use a bulk FCC reference to fill the vacuum with fictitious sites, or use the defective first frame as a defect-free WS reference. A quantitative WS slab analysis needs a separately checked matching 448-site slab reference and a clear treatment of surface relaxation; it is outside this introductory comparison.

## Completion criteria

You can explain the role of chemical identity, compare the two effective inputs and distinguish their failure stages. Preserve both inputs and logs. Clean basin-exit behavior and quantitative alloy kinetics remain unresolved.
