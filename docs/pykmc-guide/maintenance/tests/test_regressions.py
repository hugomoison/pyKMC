"""Regression fixtures: never run MPI, Slurm, LAMMPS or KMC calculations."""
import argparse
import ast
import configparser
import contextlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/_shared"))
from basin_checks import log_summary, verify_nicr_mapping
from preflight import RECIPES, load_config_class

COUNT = 0


def expect(ok, message):
    global COUNT
    if not ok:
        raise AssertionError(message)
    COUNT += 1


def run(args, **kwargs):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=45, **kwargs)


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def edit_input(directory, edits):
    path = directory / "input.in"
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve keys such as T in the installed parser.
    config.read(path)
    names = {name.lower(): name for name in config.sections()}
    for section, key, value in edits:
        if value is None:
            config[names[section]].pop(key, None)
        else:
            config[names[section]][key] = value
    with path.open("w") as handle:
        config.write(handle)


def markdown_ids(text):
    """Heading slugs plus explicit legacy anchors; exclude fenced examples."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    ids, seen = [], set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M):
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        base = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        candidate, suffix = base, 0
        while candidate in seen:
            suffix += 1
            candidate = f"{base}-{suffix}"
        seen.add(candidate)
        ids.append(candidate)
    ids.extend(re.findall(r'<a\s+id="([^"]+)"', text))
    return ids


def check_static():
    collision = markdown_ids('### B4. Activate and verify\n<a id="b4-activate-and-verify"></a>')
    expect(len(collision) != len(set(collision)), "Missed legacy/heading anchor collision")
    expect(markdown_ids('## Repeat\n## Repeat') == ['repeat', 'repeat-1'], "Repeated heading slug")
    for path in ROOT.rglob("*"):
        if path.suffix in (".sh", ".sbatch"):
            result = run(["bash", "-n", str(path)])
            expect(result.returncode == 0, result.stdout)
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        if path.suffix != ".md":
            continue
        ids = markdown_ids(path.read_text())
        expect(len(ids) == len(set(ids)), f"Duplicate rendered anchor: {path}")
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if re.match(r"\w+://|mailto:", target):
                continue
            name, _, fragment = unquote(target.strip("<>")).partition("#")
            dest = (path.parent / name).resolve() if name else path
            expect(dest.exists(), f"Missing link: {path} -> {target}")
            if fragment:
                expect(fragment in markdown_ids(dest.read_text()), f"Missing anchor: {path} -> {target}")
        for block in re.findall(r"```bash\n(.*?)\n```", path.read_text(), flags=re.S):
            expect(run(["bash", "-n"], input=block).returncode == 0, f"Invalid shell block in {path}")


def check_diagnostics(work):
    directory = work / "basin"
    directory.mkdir()
    (directory / "input.in").write_text("[Control]\nbasin = True\n")
    data = np.array([[1, 1, 1, 0, .1, 1, 1, -1, 0, 0]])
    (directory / "pykmc.events").write_text("Basin Exit Events (Selected=0)\n")
    for message, warning in (("Exploring the Basin", False), ("Exploring superbasin", True), ("", True)):
        (directory / "pykmc.log").write_text(message + "\nEnd of simulation\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            log_summary(directory, data)
        expect(("no recognized basin-attempt markers" in output.getvalue()) == warning,
               output.getvalue())
    (directory / "input.in").write_text("[Control]\nbasin = False\n")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        log_summary(directory, data)
    expect("no recognized basin-attempt markers" not in output.getvalue(), output.getvalue())
    (directory / "input.in").write_text("[Control]\n")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        log_summary(directory, data)
    expect("basin is not explicit" in output.getvalue(), output.getvalue())

    log = directory / "lammps.log.0"
    normal = ("create_box 2 box\nmass 1 51.9961\nmass 2 58.6934\n"
              "labelmap atom 1 Cr 2 Ni\npair_style eam/alloy\n"
              "pair_coeff * * ./Bonny_2013_NiFeCr.eam Cr Ni\n")
    reversed_types = normal.replace("mass 1 51.9961\nmass 2 58.6934", "mass 1 58.6934\nmass 2 51.9961").replace(
        "1 Cr 2 Ni", "1 Ni 2 Cr")
    cases = [(normal, True), (normal + normal, True),
             (normal.replace(".eam Cr Ni", ".eam Ni Cr"), False),
             (normal.replace("mass 1 51.9961", "mass 1 58.6934"), False),
             (normal.replace("labelmap atom 1 Cr 2 Ni\n", ""), False),
             (reversed_types, False),
             (reversed_types.replace(".eam Cr Ni", ".eam Ni Cr"), True),
             (normal.replace("mass 1 51.9961", "mass 1 nan"), False),
             (normal + reversed_types, False), ("", False)]
    for text, valid in cases:
        log.write_text(text)
        try:
            verify_nicr_mapping(log)
            accepted = True
        except ValueError:
            accepted = False
        expect(accepted == valid, f"Incorrect mapping result for {text}")


def check_routes(work, install):
    fakebin = work / "bin"
    fakebin.mkdir()
    fake_modules = work / "modules"
    fake_modules.mkdir()
    (fake_modules / "lammps.py").write_text(
        "class lammps:\n    def __init__(self, **kw): pass\n"
        "    def has_style(self, *args): return True\n    def close(self): pass\n")
    write_executable(fakebin / "python", f'''#!/bin/bash
if [[ "$1" == */record_versions.py ]]; then
    mkdir "$2"
    exit "${{FIXTURE_RECORD_EXIT:-0}}"
fi
exec "{sys.executable}" "$@"
''')
    for name in ("mpirun", "srun"):
        write_executable(fakebin / name, '#!/bin/bash\nprintf "FIXTURE launch only\\n"\nexit "${FIXTURE_EXIT:-0}"\n')
    write_executable(fakebin / "module", "#!/bin/bash\nexit 0\n")
    write_executable(fakebin / "sbatch", "#!/bin/bash\nprintf 'FIXTURE submitted\\n'\n")
    write_executable(fakebin / "git", "#!/bin/bash\nprintf 'fixture revision\\n'\n")
    cluster = work / "cluster install"
    cluster.mkdir()
    (cluster / "activate.sh").write_text(":\n")
    env = dict(os.environ, PATH=str(fakebin) + ":" + os.environ["PATH"],
               PYTHONPATH=str(fake_modules), PYTHONDONTWRITEBYTECODE="1", VIRTUAL_ENV=sys.prefix)
    env.pop("SLURM_JOB_ID", None)
    env.pop("GUIDE_NI_EXAMPLE", None)
    for example, (_, potential, species, basin, av) in RECIPES.items():
        base = ROOT / "examples" / example
        prepared = work / f"{example} prepared"
        prepare = ["bash", str(base / "scripts/prepare_example.sh"), str(install), str(prepared)]
        result = run(prepare, env=env)
        expect(result.returncode == 0, result.stdout)
        expected = {"input.in", "input.in.upstream", "initial_config.xyz", potential, "guide-preflight.py"}
        expect({p.name for p in prepared.iterdir()} == expected, "Unexpected prepared files")
        expect((prepared / "input.in").read_bytes() == (prepared / "input.in.upstream").read_bytes(), "Input edited")
        expect((prepared / "guide-preflight.py").read_bytes() == (ROOT / "examples/_shared/preflight.py").read_bytes(), "Wrong helper copy")
        expect(run(prepare, env=env).returncode != 0, "Existing destination accepted")
        edits = [("control", "initial_config", "./initial_config.xyz"),
                 ("control", "n_sessions", "7"), ("control", "group_size", "-1"),
                 ("control", "engine_use_rank_0", "False"), ("control", "basin", str(basin)),
                 ("control", "active_volume", str(av)), ("control", "recycle", "False"),
                 ("lammps", "pair_coeff", f"* * ./{potential} {' '.join(species)}")]
        if example == "fe-bivacancy":
            edits.append(("atomicenvironment", "rnei", "3.2"))
        edit_input(prepared, edits)
        # Execute the documented pre-submit chain, never the real scheduler.
        chapter = (base / "README.md").read_text()
        submission = re.findall(r"python -B guide-preflight.py --example [a-z-]+ &&\nsbatch job.sbatch", chapter)
        expect(len(submission) == (2 if example == "nicr-slab" else 1), "Missing guarded submission command")
        for command in submission:
            expect(f"--example {example} &&" in command, "Wrong chapter in submission preflight")
        result = run(["bash", "-c", submission[0]], cwd=prepared, env=env)
        expect(result.returncode == 0 and "PASS:" in result.stdout
               and "FIXTURE submitted" in result.stdout, result.stdout)
        rejected = work / f"{example} pre-submit invalid"
        shutil.copytree(prepared, rejected)
        edit_input(rejected, [("control", "n_sessions", "31")])
        result = run(["bash", "-c", submission[0]], cwd=rejected, env=env)
        expect(result.returncode != 0 and "FAIL: preflight" in result.stdout
               and "FIXTURE submitted" not in result.stdout, "Invalid input reached sbatch")
        if example == "nicr-slab":
            grey_submit = work / "nicr grey pre-submit"
            shutil.copytree(prepared, grey_submit)
            edit_input(grey_submit, [("atomicenvironment", "atom_coloring_mode", "grey")])
            result = run(["bash", "-c", submission[1]], cwd=grey_submit, env=env)
            expect(result.returncode == 0 and "FIXTURE submitted" in result.stdout, result.stdout)
        batch = work / f"{example}.sbatch"
        batch.write_text((base / f"trillium/{example}.sbatch").read_text().replace(
            "/scratch/YOUR_USERNAME/pyKMC_install/pykmc_install", str(cluster)))
        local = base / "local/run-local.sh"

        def launch(dest, route, **extra):
            launch_env = dict(env, **extra)
            if route == "cluster":
                launch_env.update(SLURM_SUBMIT_DIR=str(dest), SLURM_JOB_ID="fixture", SLURM_NTASKS="8")
            return run(["bash", str(batch if route == "cluster" else local)], cwd=dest, env=launch_env)

        for route, timer, code in (("local", "auto", 0), ("local", "bash", 0), ("local", "auto", 7),
                                   ("local", "bash", 7), ("cluster", "auto", 0), ("cluster", "auto", 7)):
            dest = work / f"{example} {route} {timer} {code}"
            shutil.copytree(prepared, dest)
            result = launch(dest, route, PYKMC_TIMER=timer, FIXTURE_EXIT=str(code))
            expect(result.returncode == code, f"{example} {route}: {result.stdout}")
            out = dest / "run-fixture" if route == "cluster" else dest
            expect((out / "run-exit-status.txt").read_text().strip() == str(code), "Exit status lost")
            for filename in expected:
                expect((out / filename).read_bytes() == (prepared / filename).read_bytes(), f"Changed {filename}")
            if route == "local":
                expect("real " in (out / "run.log").read_text(), "Missing timer")
            expect(launch(dest, route).returncode != 0, "Existing output accepted")

        invalid = [[("control", "n_sessions", "31")], [("control", "group_size", "8")],
                   [("control", "engine_use_rank_0", "True")], [("control", "basin", str(not basin))],
                   [("control", "active_volume", str(not av))], [("control", "recycle", "True")],
                   [("control", "trajectory_output", "../trajkmc.xyz")],
                   [("control", "reference_table", "reference_table.pickle")],
                   [("control", "initial_config", "../initial_config.xyz")],
                   [("control", "initial_config", str(prepared / "initial_config.xyz"))],
                   [("control", "trajectory_output", str(prepared / "trajkmc.xyz"))],
                   [("control", "engine", None)], [("control", "n_sessions", "7 # workers")],
                   [("lammps", "pair_coeff", f"* * {potential} WRONG")]]
        if example == "fe-bivacancy":
            invalid.append([("atomicenvironment", "rnei", "2.8")])
        if example == "nicr-slab":
            invalid.append([("lammps", "pair_coeff", f"* * {potential} Ni Cr")])
        for index, edits in enumerate(invalid):
            for route in ("local", "cluster"):
                dest = work / f"{example} invalid {index} {route}"
                shutil.copytree(prepared, dest)
                edit_input(dest, edits)
                result = launch(dest, route)
                expect(result.returncode != 0 and "FAIL: preflight" in result.stdout, result.stdout)
                expect("FIXTURE launch" not in result.stdout and not (dest / "run.log").exists()
                       and not (dest / "run-fixture").exists(), "Invalid input reached launch")
        for route in ("local", "cluster"):
            dest = work / f"{example} missing helper {route}"
            shutil.copytree(prepared, dest)
            (dest / "guide-preflight.py").unlink()
            result = launch(dest, route)
            expect(result.returncode != 0 and "Missing guide-preflight.py" in result.stdout, result.stdout)
        # Optional omitted settings must use pyKMC's actual defaults, on both routes.
        for route in ("local", "cluster"):
            dest = work / f"{example} defaults {route}"
            shutil.copytree(prepared, dest)
            edit_input(dest, [("control", "group_size", None), ("control", "engine_use_rank_0", None),
                              ("control", "recycle", None)])
            result = launch(dest, route)
            expect(result.returncode == 0, result.stdout)
        dest = work / f"{example} version record fails"
        shutil.copytree(prepared, dest)
        result = launch(dest, "local", FIXTURE_RECORD_EXIT="6")
        expect(result.returncode == 6 and "FIXTURE launch" not in result.stdout, result.stdout)
        expect((dest / "run-exit-status.txt").read_text().strip() == "6", "Lost recorder exit status")
        if example == "nicr-slab":
            for route in ("local", "cluster"):
                dest = work / f"{example} grey {route}"
                shutil.copytree(prepared, dest)
                edit_input(dest, [("atomicenvironment", "atom_coloring_mode", "grey")])
                result = launch(dest, route)
                expect(result.returncode == 0, result.stdout)
        print(f"PASS: {example} shared checks, launch protections and exit codes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", type=Path, required=True)
    args = parser.parse_args()
    check_static()
    load_config_class()
    expect("mpi4py.MPI" not in sys.modules and "pykmc" not in sys.modules,
           "Configuration-only loading unexpectedly imported MPI/pyKMC initializer")
    with tempfile.TemporaryDirectory(prefix="BG checks with spaces '") as tmp:
        work = Path(tmp)
        check_diagnostics(work)
        check_routes(work, args.install_root)
    print(f"PASS: {COUNT} assertions; simulation/engine commands were stubs, no calculation launched.")


if __name__ == "__main__":
    main()
