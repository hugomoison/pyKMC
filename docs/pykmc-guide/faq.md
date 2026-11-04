# Common questions

## Can I keep zsh?

Yes. `bash path/to/run-local.sh` uses Bash only to run that script. Your interactive terminal stays in zsh. The preparation script copies files; the launcher starts pyKMC. Use `sbatch job.sbatch` for a Trillium job.

## Which programs do I need?

The installation builds pyKMC and its Python/compiled dependencies, including LAMMPS, pARTn and IRA. OVITO Basic is a separate local application for visualization. LAMMPS-GUI is optional and is not used in these exercises.

An EAM or Stillinger–Weber file defines an interatomic potential for energies and forces. It serves a different purpose from a DFT pseudopotential. An `initial_magmoms` column in a structure file does not turn an ordinary EAM calculation into spin dynamics.

## Do I need to generate a perfect reference?

For the supplied bulk examples, load the chapter's reference XYZ directly in OVITO. It belongs to the analysis, while `initial_config.xyz` is the defective structure used for simulation. Match the material, cell and alignment.

[Reference generators](maintenance/reference-generation/README.md) are maintenance tools for rebuilding these assets. NiCr uses a slab-specific visualization workflow with actual atoms; a bulk reference would incorrectly place ideal sites in the vacuum.

## What if I stop a run?

After Ctrl+C, wait for MPI processes to terminate and check your process monitor. Keep the interrupted directory and logs. Use a new directory for a new run; copying a restart file does not by itself configure a continuation. Checkpoint continuation is outside these introductory exercises.

For Slurm jobs, an SSH disconnection leaves the submitted job running. Check `squeue` and `sacct` before submitting again. Cancellation or timeout may prevent `run-exit-status.txt` from being written, so the final Slurm state is essential.

## Which records should I keep?

Keep inputs, the potential, outputs, software versions and any source patches with the run. Local launchers write `run-versions/`; Slurm scripts record version information in `run-JOBID/`. Check warnings that a source record is unavailable. Version records do not archive compiled libraries or the contents of untracked source files.

Review records before publishing: they may contain personal paths or modified source. Scratch is temporary working storage; copy important results to backed-up storage.

## Why do I see failure messages even when the run completes?

An individual saddle search can fail while other searches produce usable events. Judge the result using the overall exit status, accepted events, refinement/reconstruction diagnostics and trajectory checks. The chapters explain which messages need investigation for each system.

## Is a successful run a validated material prediction?

A completed run with consistent output establishes that the exercise worked at that level. Diffusion coefficients, lifetimes and comparisons with experiment need additional sampling, event-catalog and potential validation. The [validation records](maintenance/validation.md) separate those claims.
