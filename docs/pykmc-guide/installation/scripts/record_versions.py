#!/usr/bin/env python3
"""Save private run provenance without changing the installed software."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New provenance directory")
    args = parser.parse_args()
    spec = importlib.util.find_spec("pykmc")
    if spec is None or not spec.origin:
        raise RuntimeError("Cannot locate the installed pyKMC package; check activation")
    # The tutorial installers keep the venv and four source repositories together.
    root = Path(sys.prefix).resolve().parent
    sources = {
        "pyKMC": Path(spec.origin).resolve().parent.parent,
        "lammps": root / "lammps",
        "IterativeRotationsAssignments": root / "IterativeRotationsAssignments",
        "artn-plugin": root / "artn-plugin",
    }
    args.output.mkdir()  # Refuse existing files/directories, including symlinks.
    report = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "architecture": platform.machine(),
        "pykmc_module": spec.origin,
        "mpi_launcher": shutil.which("mpirun"),
        "packages": sorted(
            f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
            for dist in importlib.metadata.distributions()
        ),
        "sources": {},
    }
    for name, repo in sources.items():
        entry = {"path": str(repo)}
        try:
            # Do not mistake an enclosing project repository for this dependency.
            if not (repo / ".git").exists():
                raise ValueError("Source checkout not found at the expected location")
            entry["commit"] = git(repo, "rev-parse", "HEAD").strip()
            entry["branch"] = git(repo, "branch", "--show-current").strip() or "detached HEAD"
            entry["status"] = git(repo, "status", "--short", "--untracked-files=normal")
            for suffix, options in [("working", []), ("staged", ["--cached"])]:
                patch = git(repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", *options)
                with (args.output / f"{name}-{suffix}.patch").open("x") as handle:
                    handle.write(patch)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            entry["unavailable"] = str(exc)
            print(f"WARNING: {name} source record unavailable; inspect run-versions/versions.json")
        report["sources"][name] = entry
    with (args.output / "versions.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved private software records to {args.output}/")


if __name__ == "__main__":
    main()
