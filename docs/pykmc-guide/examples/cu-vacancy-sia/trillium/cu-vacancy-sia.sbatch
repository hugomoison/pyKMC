#!/bin/bash
# EDIT THREE VALUES: --account, --mail-user, and PYKMC_ROOT.
# Submit from the prepared directory with sbatch, not bash.
#SBATCH --job-name=Cu-VSIA
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.12.4 mpi4py/4.1.0 cmake/3.31.0
PYKMC_ROOT="/scratch/YOUR_USERNAME/pyKMC_install/pykmc_install"
source "$PYKMC_ROOT/activate.sh"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:?Submit from the prepared example directory}"
[[ "${SLURM_NTASKS:?Missing Slurm task count}" == 8 ]] || {
    printf 'This template requires eight tasks and seven worker sessions.\n' >&2
    exit 2
}
for name in input.in initial_config.xyz Cu01.eam.alloy; do
    [[ -s "$name" ]] || { printf 'Missing input: %s\n' "$name" >&2; exit 2; }
done

# Use the same prepared input checker as the local route (no MPI initialization).
[[ -s guide-preflight.py ]] || {
    printf 'Missing guide-preflight.py; prepare a fresh directory with the current guide.\n' >&2
    exit 2
}
python -B guide-preflight.py --example cu-vacancy-sia

run_dir="$PWD/run-${SLURM_JOB_ID:?Missing Slurm job ID}"
mkdir "$run_dir"
cp input.in initial_config.xyz Cu01.eam.alloy "$run_dir/"
cp guide-preflight.py "$run_dir/"
if [[ -f input.in.upstream ]]; then cp input.in.upstream "$run_dir/"; fi
cd "$run_dir"

# Preserve the launch status, including normal error exits. Slurm remains the
# authority for cancellation/timeouts (an immediate kill can prevent this trap).
record_exit() {
    run_status=$?
    trap - EXIT
    printf '%s\n' "$run_status" > run-exit-status.txt
    printf 'Run exit status: %s\n' "$run_status"
    exit "$run_status"
}
trap record_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python --version > python-version.txt 2>&1
module list > modules.txt 2>&1
git -C "$PYKMC_ROOT/pyKMC" rev-parse HEAD > pykmc-commit.txt
git -C "$PYKMC_ROOT/pyKMC" status --short > pykmc-source-status.txt

srun --ntasks="$SLURM_NTASKS" \
     --distribution=block:block --cpu-bind=cores --mem-bind=local \
     --kill-on-bad-exit=1 python -u -m pykmc -in input.in
