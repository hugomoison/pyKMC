# Paths and terminal basics

You can follow the examples by entering directories with `cd` and typing paths directly. Defining `PYKMC_ROOT`, `GUIDE_DIR` or `RUN_DIR` is optional.

## Enter a directory with Tab completion

In Bash and zsh, `~` means your home directory. Type the first part of a directory name and press Tab to complete it; if several names match, display the choices and type a few more letters.

For example, start with `cd ~/Research/` and complete the remaining folders on your own computer. Use the ordinary ASCII tilde `~`, not the full-width character `～`.

```bash
pwd
ls
cd ..
```

These commands show your current directory, list its contents, and move up one level. Opening a folder in Finder or File Explorer does not change the terminal's directory.

Paths with spaces need quoting or escaping. For example:

```bash
cd ~/Research/"My Projects"/pykmc-guide
```

Keep `~` outside the quotes so the shell expands it. Tab completion normally handles the necessary escaping for you.

## Suggested locations

For a first setup, place the complete guide at `~/pykmc-guide` as described on the [home page](README.md#before-copying-commands). The tutorial commands use the locations below. An existing guide or installation can remain elsewhere: substitute its path with Tab completion. Renaming or moving an installed Python environment can break it; the guide folder is separate from that environment.

| Content | Local computer | Trillium |
| --- | --- | --- |
| Complete guide | `~/pykmc-guide` | `~/pykmc-guide` |
| pyKMC installation | `~/software/pyKMC_setup/pykmc_install` | `"$SCRATCH/pyKMC_install/pykmc_install"` |
| Runs | `~/pykmc-runs/example-01` | `"$SCRATCH/pykmc-runs/example-01"` |

The installation directory contains both `pykmc_env/` and the `pyKMC/` source checkout. On Windows, use paths inside Ubuntu for computing. `$SCRATCH` is a path supplied by Trillium; it is already set when you log in.

## What preparation creates

From an example directory, a command such as:

```bash
bash scripts/prepare_example.sh \
  ~/software/pyKMC_setup/pykmc_install \
  ~/pykmc-runs/ni-monovacancy-01
```

has two arguments: the installation directory to read, and the new run directory to create. The destination determines where the copied files go, regardless of where you keep the script.

The new directory contains:

- `input.in`: the settings you review and edit.
- `initial_config.xyz`: the starting atomic configuration.
- The example's potential file.
- `input.in.upstream`: an unchanged input backup.
- `guide-preflight.py`: the automatic input checker used at launch.

Let preparation create the destination. It refuses an existing directory to protect previous results.

For local runs, enter that directory and launch the script from its location in the guide. The script and inputs can stay in separate places. For Trillium, the prepared directory is the **submission directory**; each job creates a **result directory** named `run-JOBID/` inside it. Slurm logs remain in the submission directory.

## Returning to a result

Activate the appropriate environment in the new terminal, then `cd` into the existing result directory and use the chapter's checks. Preparation and launch are only for starting a new run.

On Trillium, use the existing `run-JOBID/` directory. For local analysis of downloaded results, activate your local environment.

## Optional: abbreviate repeated paths

If you prefer short names, shell variables can hold paths:

```bash
GUIDE_DIR=~/pykmc-guide/examples/ni-monovacancy
RUN_DIR=~/pykmc-runs/ni-monovacancy-01
cd "$RUN_DIR" &&
bash "$GUIDE_DIR/local/run-local.sh"
```

An assignment stores a path; `cd` enters it. Keep quotes around later `"$VARIABLE"` expressions, and set the variables again in each new terminal. Scripts may use variables internally even when the tutorial commands use direct paths.
