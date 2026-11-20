# First publication checklist

The first edition can be submitted for review while parameter calibration continues. Keep completed demonstrations and unresolved diagnostics clearly distinguished in the chapters and [validation record](validation.md).

## Repository and folder

Use `pykmc-guide` as the guide folder name. The first edition is proposed under `docs/pykmc-guide/`, on branch `docs/beginner-guide-develop` in the [contributor fork](https://github.com/yihangliu-research/pyKMC/tree/docs/beginner-guide-develop), targeting upstream `develop`. Maintainers can review the location and website integration in the PR. Keep the installed pyKMC checkout and simulation directories outside this publishing workflow.

A repository name and its local guide-directory name can differ. The home page links to the review branch's ZIP and explains how to copy its complete `docs/pykmc-guide/` folder to `~/pykmc-guide`. After upstream merges the guide, update that link to the accepted upstream branch. Guide downloads and software installations remain separate.

## Review before committing

1. Run the [regression tests](tests/README.md) and review the rendered home page, installation anchors and first example. Check the expanded local and Trillium blocks, including the pre-submit input check.
2. Inspect the file list and diff. Include reusable documentation, scripts, reference lattices and sanitized illustrations. Exclude installations, run outputs, personal paths, emails, job IDs, credentials, `run-versions/` and private patch records.
3. Check attribution and licensing with the upstream maintainers, including the included recovery patch and screenshots. Do not assign a new license to third-party content without authority.
4. Keep changes on a guide branch. Describe tested platforms, fixture tests and outstanding Fe/NiCr/WSL limitations in the PR; do not claim that all calculations are scientifically validated.
5. After review, make follow-up commits on the same branch and push them to update the PR. Upstream maintainers decide when and how to merge.

From the publishing checkout, inspect before staging:

```bash
git status --short
git diff --check
git diff
```

Stage only the agreed guide directory. Then inspect `git diff --cached --stat` and `git diff --cached` before committing. The guide's `.gitignore` reduces accidental inclusion of run files, but ignored patterns do not remove files already tracked by Git.

## Naming rules

- Example and documentation directories: lowercase, hyphen-separated names.
- Chapter entry pages: `README.md`; other documentation: lowercase `.md` names.
- Python modules/checkers and `prepare_example.sh`: keep the existing underscore convention and working references.
- Local launcher: `run-local.sh`; Slurm script: the example's existing hyphen-separated `.sbatch` name.
- Potentials and reference lattices: preserve their established scientific filenames. Avoid cosmetic renames that break commands or provenance.

## Updating calibrated parameters

Review the input notes, installed input source, shared preflight constraints and validation record together. Rerun script tests and the affected scientific example before updating its validation status. Keep documentation-only improvements separate from numerical retuning.
