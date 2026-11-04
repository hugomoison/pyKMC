#!/bin/bash
# Run from a NEW Ni SIA working directory with the pyKMC venv active.
# Usage: bash /path/to/examples/ni-fcc-sia/local/run-local.sh
# Eight-rank launch for this example, not a scheduler or an installer.

set -o pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf 'Use the Trillium batch recipe inside a Slurm allocation.\n' >&2
    exit 2
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    printf 'Activate your local pyKMC virtual environment first.\n' >&2
    exit 2
fi
for required_command in python mpirun tee; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$required_command" >&2
        exit 2
    fi
done
case "${PYKMC_TIMER:-auto}" in
    auto|bash) ;;
    *) printf 'PYKMC_TIMER must be auto or bash.\n' >&2; exit 2 ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
recorder="$script_dir/../../../installation/scripts/record_versions.py"
if [[ ! -f "$recorder" ]]; then
    printf 'Keep the full guide together; missing version recorder: %s\n' "$recorder" >&2
    exit 2
fi
for input_file in input.in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam; do
    if [[ ! -s "$input_file" ]]; then
        printf 'Missing or empty input: %s (working directory: %s)\n' "$input_file" "$PWD" >&2
        exit 2
    fi
done
# Never reuse a result directory. These patterns cover this tutorial's outputs.
for output_file in run.log run-exit-status.txt run-versions pykmc.* trajkmc.xyz restart_*.npz \
                   reference_table.pickle visited_environments.pickle artn.out.* lammps.log.*; do
    if [[ -e "$output_file" || -L "$output_file" ]]; then
        printf 'Existing output: %s. Prepare a fresh run directory; nothing was removed.\n' "$output_file" >&2
        exit 2
    fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Shared input checks; preparation snapshots this helper for both routes.
if [[ ! -s guide-preflight.py ]]; then
    printf 'Missing guide-preflight.py; prepare a fresh directory with the current guide.\n' >&2
    exit 2
fi
python -B guide-preflight.py --example "${GUIDE_NI_EXAMPLE:-ni-fcc-sia}" --check-lammps
preflight_status=$?
if [[ "$preflight_status" -ne 0 ]]; then
    exit "$preflight_status"
fi

# Atomically claim a fresh log; a simultaneous second invocation must stop.
if ! (set -o noclobber; : > run.log); then
    printf 'Cannot create a fresh run.log; stop and inspect this directory.\n' >&2
    exit 2
fi

# Preserve versions and tracked patches before starting MPI.
python "$recorder" run-versions 2>&1 | tee -a run.log
record_status=$?
if [[ "$record_status" -ne 0 ]]; then
    printf 'Version recording failed; no simulation was started.\n' | tee -a run.log
    printf '%s\n' "$record_status" > run-exit-status.txt
    exit "$record_status"
fi

run_timed() {
    if [[ "${PYKMC_TIMER:-auto}" == auto && -x /usr/bin/time ]]; then
        /usr/bin/time -p mpirun -n 8 python -u -m pykmc -in input.in
    else
        # Bash's timer needs no extra Ubuntu/WSL package. The group is piped to tee.
        TIMEFORMAT=$'real %3R\nuser %3U\nsys %3S'
        time mpirun -n 8 python -u -m pykmc -in input.in
    fi
}
run_timed 2>&1 | tee -a run.log

run_status=$?
printf 'Run exit status: %s\n' "$run_status" | tee -a run.log
printf '%s\n' "$run_status" > run-exit-status.txt
exit "$run_status"
