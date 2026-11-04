# Targeted helper regression tests

Run with the installed pyKMC environment active, from any directory:

```bash
python /absolute/path/to/guide/maintenance/tests/test_regressions.py \
  --install-root /absolute/path/to/pykmc_install
```

The tests use the installed examples and configuration parser, but replace MPI,
Slurm, version-recording and LAMMPS-loading commands with fixtures. Temporary
directories include spaces and an apostrophe. No KMC calculation, scheduler
submission, minimization, installation change or write to existing run data is
performed. LAMMPS availability remains a separate installation check.

Coverage: preparation copies/overwrite refusal; pre-submit input checks that block
the scheduler stub on failure (including the NiCr grey command); both launchers' identical input
acceptance; group/session layout; malformed/unsupported inputs; missing helper;
exit propagation and both local timers; existing-output protection; changed basin
wording; NiCr label/mass/EAM agreement and missing evidence; guide links, duplicate
heading/legacy anchors and command syntax.

The tests require the current guide's full directory structure and the Python
dependencies already used by its checkers. Historical trajectory checks are not
bundled: use the chapter's checker on your own saved results.
