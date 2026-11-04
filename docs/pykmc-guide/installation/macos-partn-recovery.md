# macOS recovery: pARTn C-interface symbols are not exported

This is a **conditional troubleshooting appendix**, not a mandatory step for every user. Return to [Installation](README.md) if your installation already passes its checks.

## 1. Confirm that this is the same problem

Use this appendix only if LAMMPS and IRA built successfully but pARTn stops at:

```text
AttributeError: dlsym(..., artn_create): symbol not found
[FAIL] pypARTn not working
```

This patch targets pARTn revision `edea36aca8215a1d484b3b8695ecb9676fe56498`.
Some Apple Silicon/Homebrew Fortran builds do not export C-binding wrappers in
private modules from the dynamic library.
The fix explicitly makes these wrappers public. It changes interface visibility,
not the event-search algorithm, potential or simulation parameters.

The accompanying [patch](patches/partn-public-c-bindings-edea36a.patch) contains
the six-file, thirteen-line interface-export change.
This is a local workaround awaiting maintainer review, not an upstream release.
It is **not** a general solution for wrong library paths, architecture mismatch,
missing dependencies or every `symbol not found` error.

Do not delete the partial installation. Do not rerun the full installer, replace
the Python environment, or apply this patch to an already working installation.

## 2. Locate the installation and supplied patch

The following uses the generic installation directory from the installation chapter. Change it if
you chose another location. Use a local zsh or Bash terminal:

```bash
PYKMC_ROOT="$HOME/software/pyKMC_setup/pykmc_install"
source "$PYKMC_ROOT/pykmc_env/bin/activate"
unset PYTHONPATH
git -C "$PYKMC_ROOT/artn-plugin" rev-parse HEAD
git -C "$PYKMC_ROOT/artn-plugin" status --short
```

The revision must match the one above. Stop for review if tracked files already
contain changes, if the revision differs, or if your error differs. An untracked
build/output directory alone does not mean the source has been modified.

Use the patch already included in the full guide. Set `GUIDE_ROOT` to your
downloaded guide folder; no separate download or copying is required:

```bash
GUIDE_ROOT="$HOME/pykmc-guide"
PYKMC_PATCH="$GUIDE_ROOT/installation/patches/partn-public-c-bindings-edea36a.patch"
less "$PYKMC_PATCH"
```

Press `q` to leave the viewer. The first line should start with `diff --git`.

## 3. Back up, apply and rebuild only pARTn

The block below checks the revision and tracked changes before making changes.
It preserves the six original source files and both installed pARTn libraries in
a uniquely named backup directory **outside** the pARTn repository. It does not
delete anything. Keep this backup and the repair log.

```bash
set -o pipefail
(
  set -eu
  test -f "$PYKMC_PATCH"
  test "$(git -C "$PYKMC_ROOT/artn-plugin" rev-parse HEAD)" = \
    edea36aca8215a1d484b3b8695ecb9676fe56498
  git -C "$PYKMC_ROOT/artn-plugin" diff --quiet
  git -C "$PYKMC_ROOT/artn-plugin" diff --cached --quiet
  git -C "$PYKMC_ROOT/artn-plugin" apply --check "$PYKMC_PATCH"

  PARTN_LIB_DIR=$(python -c 'import pathlib, pypARTn; print(pathlib.Path(pypARTn.__file__).resolve().parent / "lib")')
  test "$PARTN_LIB_DIR" = "$PYKMC_ROOT/pykmc_env/lib/python3.13/site-packages/pypARTn/lib"
  PARTN_BACKUP=$(mktemp -d "$PYKMC_ROOT/partn-backup.XXXXXX")
  mkdir -p "$PARTN_BACKUP/src" "$PARTN_BACKUP/lib"
  for name in artn_api.f90 artn_step.f90 d_datainfo.f90 h_artn_info.f90 h_artn_info.f90.in m_artn_fire.f90; do
    cp "$PYKMC_ROOT/artn-plugin/src/$name" "$PARTN_BACKUP/src/"
  done
  cp "$PARTN_LIB_DIR/libartn.dylib" "$PARTN_BACKUP/lib/"
  cp "$PARTN_LIB_DIR/libartn-lmp.dylib" "$PARTN_BACKUP/lib/"
  printf 'Backup: %s\n' "$PARTN_BACKUP"

  git -C "$PYKMC_ROOT/artn-plugin" apply "$PYKMC_PATCH"
  cmake --build "$PYKMC_ROOT/artn-plugin/build" --parallel 4
  cmake --install "$PYKMC_ROOT/artn-plugin/build"
  git -C "$PYKMC_ROOT/artn-plugin" diff --stat
) 2>&1 | tee "$PYKMC_ROOT/partn-repair.log"
echo "Repair exit status: $?"
```

Expect exit status 0 and six changed source files with thirteen added lines. If
any check/build fails, stop and preserve the log; do not force the patch or
continue to a simulation. If the source patch applied but compilation failed,
rerunning the whole block will stop at its clean-source check. Diagnose the build
error rather than applying the patch twice.

This block assumes the Python 3.13 directory layout from the installation chapter and a normal
non-symlinked installation path. If the library-path guard fails, inspect the
printed interpreter/package paths rather than removing the guard blindly.

## 4. Verify and resume

Return to the installation chapter's local activation and one-rank library check, then follow the
[local Ni monovacancy workflow](../examples/ni-monovacancy/local/README.md) in a separate
directory. A build success message alone does not prove Python/MPI/plugin integration.

If the original installer stopped before generating top-level `activate.sh`,
that file may not exist. It is **not necessary** for the workflow in this guide:
directly sourcing `pykmc_env/bin/activate` followed by `unset PYTHONPATH` works in
both zsh and Bash.

Record the patch with the source commits and test results. Updating or replacing
pARTn may overwrite it; do not assume the patch remains necessary or applicable
after an upstream update. Ask the maintainers to review the fix before making it
part of the project's standard installer.
