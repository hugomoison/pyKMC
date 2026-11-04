# Ni FCC self-interstitial

Goal: run 10 KMC steps and connect self-interstitial motion to barriers and rates.

Prerequisite: [Installation](../../installation/README.md); [Ni monovacancy](../ni-monovacancy/README.md) is the recommended first exercise.

## 1. Understand the system

A [self-interstitial atom (SIA)](../../glossary.md#vacancy-sia-and-dumbbell) is an extra host atom. This configuration has 4001 Ni atoms and 4000 ideal FCC sites in a periodic 35.2 Å cube. In a dumbbell configuration, two nearby atoms straddle one lattice site, accommodating one extra atom.

Preparation copies `initial_config.xyz`, `input.in` and `Ni_v6_2.0_LKBeland2016.eam` from the installed `examples/Ni_fcc_4001at_sia`.

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
cd ~/pykmc-guide/examples/ni-fcc-sia &&
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/ni-fcc-sia-01 &&
cd ~/pykmc-runs/ni-fcc-sia-01 &&
ls -lh input.in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam
```

The inputs are now together in `~/pykmc-runs/ni-fcc-sia-01`. Open the settings:

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
cd ~/pykmc-guide/examples/ni-fcc-sia &&
bash scripts/prepare_example.sh \
  "$SCRATCH/pyKMC_install/pykmc_install" \
  "$SCRATCH/pykmc-runs/ni-fcc-sia-01" &&
cd "$SCRATCH/pykmc-runs/ni-fcc-sia-01" &&
ls -lh input.in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam
```

This is the submission directory. Open the settings:

```bash
nano input.in
```

</details>

Preparation also saves `input.in.upstream` and `guide-preflight.py` automatically. It leaves numerical settings unchanged. Apply the input checklist below before launching. In nano, save with Ctrl+O, Enter, then exit with Ctrl+X.

<a id="1-prepare-and-understand-the-example"></a>
<a id="input"></a>

## 3. Input checklist

Example parameters are under maintainer review. Required run settings are listed below; [validation](../../maintenance/validation.md#current-input-settings) records numerical settings and source versions. Preparation copies the installed input unchanged.

### Required files and run layout

- Set `initial_config = ./initial_config.xyz`.
- Use `pair_style = eam/alloy` and `pair_coeff = * * ./Ni_v6_2.0_LKBeland2016.eam Ni`.
- Set `n_steps = 10`. This teaching length extends the supplied one-step test so you can inspect several transitions.
- Use `basin = False`, `active_volume = False` and `recycle = False`.

Use `engine = lammps`, `n_sessions = 7`, `engine_use_rank_0 = False` and the default `group_size = -1` with the supplied 8-process launcher. For a new run, leave restart and imported catalog paths unset. Set feature switches in `[Control]`; a section heading alone does not enable a feature.

### Understand the physical parameters

`cna/graph` recognizes local crystal environments. `rnei` controls the neighbor cutoff and `rcut` the environment radius; neither substitutes for the potential's interaction cutoff. `min_style` and `minimize` control structural relaxation, while PSR/IRA settings control structure matching and event reuse.

`nsearch` controls search attempts for each new environment; some attempts fail or find equivalent events. `emin_event` and `emax_event` set the accepted barrier range in eV. `k0` is the attempt-frequency prefactor in ps⁻¹ and `T` is temperature in K; with `style = constant`, the event rate is `k0 exp(-Ea/kBT)`. The prefactor is constant, while the rate depends on the barrier.

The copied input contains the numerical values for this run; no additional physical-parameter edit is required here. Use the validation record to investigate differences if your installed example has changed.

<a id="run"></a>

## 4. Run

Both routes use 8 MPI processes and 7 worker sessions. The launcher checks inputs and protects existing results. For a local run, provide 8 available MPI CPU slots; physical cores and logical threads can differ, particularly under WSL. Review the layout before using a smaller machine rather than forcing oversubscription.

<details>
<summary>Local computer — launch and check the exit</summary>

```bash
cd ~/pykmc-runs/ni-fcc-sia-01 &&
bash ~/pykmc-guide/examples/ni-fcc-sia/local/run-local.sh
```

Keep the terminal open and computer awake. On macOS, you can replace the `bash` line with `caffeinate -i bash ~/pykmc-guide/examples/ni-fcc-sia/local/run-local.sh` to prevent idle sleep; leave the lid open.

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

Copy the [Slurm script](trillium/ni-fcc-sia.sbatch) to the submission directory:

```bash
cd "$SCRATCH/pykmc-runs/ni-fcc-sia-01" &&
cp ~/pykmc-guide/examples/ni-fcc-sia/trillium/ni-fcc-sia.sbatch job.sbatch
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
python -B guide-preflight.py --example ni-fcc-sia &&
sbatch job.sbatch
```

Replace `JOBID` in the commands below with the number returned by Slurm:

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Pending means the job is waiting. After it finishes, expect `COMPLETED` / `0:0`. Inspect the Slurm logs and enter the result directory:

```bash
cd "$SCRATCH/pykmc-runs/ni-fcc-sia-01" &&
tail -n 30 slurm-JOBID.out &&
tail -n 30 slurm-JOBID.err &&
cd run-JOBID &&
cat run-exit-status.txt
```

Expect status `0`. Slurm logs stay in the submission directory; simulation outputs are in `run-JOBID/`. Run the chapter's result checks there with the same environment active.

For local OVITO, run `pwd` inside the remote result directory. Then use a separate local terminal to download it. Replace the username and remote path below with your own; choose a destination that does not exist:

```bash
mkdir -p ~/pykmc-results
test ! -e ~/pykmc-results/ni-fcc-sia-01-trillium &&
scp -r YOUR_USERNAME@trillium.alliancecan.ca:/absolute/remote/path/run-JOBID \
  ~/pykmc-results/ni-fcc-sia-01-trillium
```

Preserve `job.sbatch` and the parent Slurm logs too. To run the Python checks locally, activate local pyKMC and enter the downloaded directory. OVITO can open the downloaded trajectory directly.

</details>

If interrupted, keep the directory and logs; wait for local MPI processes to stop before another attempt. A timeout or cancellation may leave no exit-status file. See [interrupted runs](../../faq.md#what-if-i-stop-a-run) for recovery boundaries. Each new attempt uses a new directory.

<a id="2-check-completion"></a>
<a id="results"></a>

## 5. Check completion

Enter the result directory with pyKMC active, then check:


1. Local: confirm `0` in `run-exit-status.txt`; inspect `run.log` for launcher errors. Trillium: use `sacct` to confirm `COMPLETED` and `0:0`, and inspect `slurm-JOBID.out/.err` in the parent submission directory. The batch template also records `run-exit-status.txt` inside `run-JOBID/`, but does not create `run.log`.
2. For both routes, find `End of simulation` in `pykmc.log` and step 10 in `pykmc.out`. Open `pykmc.info` to check successful searches, accepted events and refinements.
3. Run the shared checker:

```bash
python ~/pykmc-guide/examples/ni-fcc-sia/scripts/check_sia_run.py trajkmc.xyz --steps 10 \
  --progress pykmc.out --input input.in \
  --reference ~/pykmc-guide/examples/ni-fcc-sia/Ni_fcc_perfect_4000_reference.xyz
```

For this fixed-cell exercise, expect 11 frames, 4001 Ni atoms per frame, finite coordinates, zero vacant reference sites and one excess atom. The checker also checks increasing physical time and the selected rates against the stated constant-prefactor model. These checks apply to this configuration and output format.

The successful checker includes:

```text
PASS: 11 frames, 4001 Ni, fixed periodic cell; zero vacancies and one excess atom per frame
PASS: progress rows, positive rates and increasing/cumulative physical time
PASS: selected-event rates agree with the Arrhenius expression
```

### Search failures are not the same as a failed run

`EIGENVALUE LOST` can terminate individual ARTn search attempts while other attempts succeed. Check successful searches, usable events, refinements, termination and structures together. If there are no valid events, the run stops, or checks fail, investigate before continuing.

Search attempts, successful searches, distinct reference events and active events are different counts. Symmetry-related channels can arise from one reference event. Exact counts and event IDs vary between runs. Some versions group several errors other than `EVENT_NOT_FOUND` under `minima_not_matching_positions`; inspect detailed diagnostics before assigning a specific cause.

<a id="3-read-barriers-rates-and-physical-time"></a>
<a id="rates"></a>

## 6. Read barriers, rates and physical time

Row `i` of `pykmc.out` describes the event taking frame i−1 to frame i. Row 0 records the initial state. Read the file's header for the installed version.

```text
Step  dT(s)  T(s)  Ref event  Ea(eV)  k_evt(ps-1)  k_tot(ps-1)  E(eV)  Cpu time(s)  Wall time(s)
```

| Column | Meaning |
| --- | --- |
| `Step` | KMC step number. |
| `dT(s)` | Sampled physical waiting time for this step. |
| `T(s)` | Cumulative simulated time; T here is not temperature. |
| `Ref event` | Selected reference-event ID, not an atom ID. |
| `Ea(eV)` | Selected forward activation barrier, not the system's total energy. |
| `k_evt(ps-1)` | Rate of the selected individual channel. |
| `k_tot(ps-1)` | Sum of rates over the currently included active channels. |
| `E(eV)` | System potential energy after the event. |
| `Cpu time(s)` | CPU time used by the process recording this step, not summed over all MPI ranks. |
| `Wall time(s)` | Elapsed computational time measured within this step. |

CPU and wall timings measure computation for each step. The launcher's whole-run timer also includes startup and shutdown; KMC physical time is reported separately in `dT(s)` and `T(s)`.

For the supplied constant-prefactor model:

```text
k_i = k0 × exp[−Ea_i / (kB × temperature)]
P_i = k_i / k_tot
mean waiting time = 1 / k_tot
sampled waiting time = −ln(u) / k_tot, with 0 < u < 1
```

Use `kB = 8.617333262 × 10⁻⁵ eV/K`. Rates in ps⁻¹ give waiting times in ps; pyKMC converts these to seconds in the output. `constant` fixes the prefactor, not every event's rate. Rate is not an atomic velocity. A longer sampled `dT` does not by itself indicate a higher selected barrier, because waiting time depends on all active rates and a random draw.

Compare two distinct candidates in the same step of `pykmc.events`: inspect forward barrier, rate, reference ID and `Refined`. With the same prefactor and temperature, a higher barrier gives a smaller per-channel rate. A class's selection probability is the sum over its channels, so multiplicity matters too.

Candidate does not mean executed. `pykmc.events` lists alternatives; `pykmc.out` describes the selected event. Ten steps may all select the same low-barrier reference type, even when higher-barrier candidates exist. Small numerical barrier differences within one type are not automatically different mechanisms. An unrefined candidate is not equivalent to a locally refined and executed event.

<a id="4-visualize-sia-position-and-dumbbell-orientation"></a>
<a id="visualization"></a>

## 7. Visualize SIA position and dumbbell orientation

Open the completed `trajkmc.xyz` in local OVITO and confirm frames 0–10. Disable old analysis modifiers. Use the supplied [4000-site reference](Ni_fcc_perfect_4000_reference.xyz) for this SIA cell.

The reference is ready to use and matches this aligned 35.2 Å FCC box.

### A. Sites: locate the defect

1. Add Wigner–Seitz defect analysis.
2. Choose External file, load the shared reference, and keep a constant reference.
3. Choose Sites, leave per-type occupancies off and affine mapping off.
4. Before filtering, check 0 vacancies and 1 interstitial.
5. Add Expression selection with `Occupancy <= 1`, then Delete selected.

One marker remains: the multiply occupied reference site, not a physical atom. An occupancy of 2 means one excess atom, not two SIAs.

### B. Atoms: inspect the actual dumbbell

Switch WS output to Atoms and keep the same selection/deletion modifiers. For dumbbell configurations, two real atoms remain. Inspect their orientation in multiple views; projection along the dumbbell can make the two spheres overlap.

The selected pair can change identity between frames. This filter finds the current occupants of an overoccupied site; it does not track a permanently fixed pair. Migration can exchange a dumbbell member with a lattice atom. The surrounding atoms also relax, although they are hidden here. For fixed-atom displacement tracking, disable WS and use the original trajectory.

Opposite box faces connect periodically; use minimum-image displacements to interpret apparent cross-box jumps. The trajectory shows event endpoints, not the continuous saddle-crossing path or thermal vibrations. See the [OVITO WS manual](https://www.ovito.org/manual/reference/pipelines/modifiers/wigner_seitz_analysis.html).

<a id="5-connect-a-visible-change-to-its-rate"></a>
<a id="interpret"></a>

## 8. Connect a visible change to its rate

Choose one clear transition from your own trajectory:

1. Compare frames i−1 and i: does the occupied site move, does the dumbbell orientation change, and do its members change?
2. Read row i of `pykmc.out`: note reference ID, barrier, selected rate and waiting time.
3. Read step i of `pykmc.events`: compare the selected channel with other candidates and their refinement status.
4. Explain what was observed versus what remained only a candidate. Do not assign an unseen high-barrier candidate a mechanism solely from its energy.

For a higher-barrier class with several channels, add their rates and divide by `k_tot` to estimate its probability of selection at that step. Failure to observe it in ten steps is not evidence it is absent. Longer sampling is optional and should serve a question; do not change temperature or selection rules merely to force a desired animation.

## Completion criteria

The run and checks pass, and you can explain one observed transition, its barrier/rate, and the distinction between a site marker and real atoms. Save your OVITO state with accessible source files. Diffusion statistics require additional sampling.
