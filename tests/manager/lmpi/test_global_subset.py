"""Regression tests: the global LAMMPS may run on a subset of workers (fewer
cores than the full engine pool) without deadlocking.

Guards against the n_sessions>=2 global/local mode-mismatch deadlock and exercises
the "few-global-of-many" topology (e.g. 100 workers, global on the first 4).

Each case runs a real MPI program in a subprocess WITH A TIMEOUT, so a deadlock
surfaces as a test failure instead of hanging the suite.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
DRIVER = os.path.join(HERE, "_mpi_drivers", "global_subset_driver.py")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
INI = os.path.join(REPO_ROOT, "tests", "data", "input.in")
MPIRUN = os.environ.get("MPIRUN", "/Users/stephenkerr/openmpi/bin/mpirun")


def _run(n_sessions, n_global_sessions, nproc, use_rank_0=False, timeout=180):
    """Run the MPI driver; return stdout. Raises subprocess.TimeoutExpired on deadlock."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        MPIRUN, "--oversubscribe", "-n", str(nproc),
        sys.executable, DRIVER,
        str(n_sessions), str(n_global_sessions), INI, str(use_rank_0).lower(),
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, timeout=timeout,
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"driver failed rc={proc.returncode}\n"
        f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return proc.stdout


def _field(stdout, key):
    for line in stdout.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{key} not found in driver output:\n{stdout}")


def _pe(stdout):
    return float(_field(stdout, "GLOBAL_PE"))


@pytest.mark.parametrize(
    "n_sessions,n_global_sessions,nproc",
    [
        (2, 1, 3),   # global on chunk-0 only (1 of 2 workers)
        (2, 2, 3),   # backward compat: global on all chunks
        (5, 2, 6),   # few-global-of-many: global on first 2 of 5 workers
    ],
)
def test_global_subset_no_deadlock(n_sessions, n_global_sessions, nproc):
    try:
        out = _run(n_sessions, n_global_sessions, nproc)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"DEADLOCK: n_sessions={n_sessions}, n_global_sessions={n_global_sessions} "
            "did not finish within timeout"
        )
    assert _pe(out) < 0  # cohesive FCC Ni potential energy is negative


def test_global_subset_uses_fewer_ranks():
    """Global comm = first n_global_sessions chunks only (the rest are local-only)."""
    out = _run(n_sessions=5, n_global_sessions=2, nproc=6)
    # chunks of [1,2,3,4,5] into 5 -> [[1],[2],[3],[4],[5]]; first 2 -> [1, 2]
    assert _field(out, "GLOBAL_RANKS") == "[1, 2]"


def test_subset_and_full_energy_agree():
    """Global energy must match within MPI-decomposition rounding regardless of core count."""
    pe_subset = _pe(_run(n_sessions=2, n_global_sessions=1, nproc=3))
    pe_full = _pe(_run(n_sessions=2, n_global_sessions=2, nproc=3))
    assert abs(pe_subset - pe_full) < 1e-5


@pytest.mark.parametrize(
    "n_sessions,n_global_sessions,nproc",
    [
        (1, 1, 2),   # rank 0 is the sole-chunk engine + orchestrator
        (2, 1, 3),   # rank 0 is engine+orchestrator AND the global subset is a subset
    ],
)
def test_use_rank_0_engine_no_deadlock(n_sessions, n_global_sessions, nproc):
    """engine_use_rank_0=True: rank 0 runs an engine in a background thread while
    also orchestrating. Regression for the boot-mode start() deadlock."""
    try:
        out = _run(n_sessions, n_global_sessions, nproc, use_rank_0=True)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"DEADLOCK: use_rank_0=True n_sessions={n_sessions}, "
            f"n_global_sessions={n_global_sessions} did not finish within timeout"
        )
    assert _pe(out) < 0
