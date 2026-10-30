#!/bin/bash
# Reuse the same eight-rank launcher; keep the caller's input/output directory.
# Keep the full guide together, and invoke this from a fresh prepared run.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
export GUIDE_NI_EXAMPLE=ni-monovacancy-sia
exec /bin/bash "$script_dir/../../ni-fcc-sia/local/run-local.sh" "$@"
