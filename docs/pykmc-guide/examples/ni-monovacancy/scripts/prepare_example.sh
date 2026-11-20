#!/bin/bash
# Copy the installed Ni monovacancy example into a NEW working directory.
# Usage: bash prepare_example.sh /path/to/pykmc_install /path/to/new-run
set -e
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash prepare_example.sh PYKMC_ROOT NEW_RUN_DIRECTORY\n' >&2
    exit 2
fi
source_example="$1/pyKMC/examples/Ni_fcc_2047at_monovacancy"
new_run_dir="$2"
if [[ -s "$source_example/input.in.template" ]]; then
    source_input="$source_example/input.in.template"
elif [[ -s "$source_example/input.in" ]]; then
    source_input="$source_example/input.in"
else
    printf 'No input.in.template or input.in found in %s\n' "$source_example" >&2
    exit 2
fi
for input_file in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam; do
    if [[ ! -s "$source_example/$input_file" ]]; then
        printf 'Missing or empty source input: %s/%s\n' "$source_example" "$input_file" >&2
        exit 2
    fi
done
if [[ -e "$new_run_dir" || -L "$new_run_dir" ]]; then
    printf 'Destination already exists: %s. Choose a new directory; nothing was changed.\n' "$new_run_dir" >&2
    exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
preflight="$script_dir/../../_shared/preflight.py"
if [[ ! -s "$preflight" ]]; then
    printf 'Keep the complete guide together; missing preflight helper: %s\n' "$preflight" >&2
    exit 2
fi
mkdir -p "$(dirname "$new_run_dir")"
mkdir "$new_run_dir"
for input_file in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam; do
    cp "$source_example/$input_file" "$new_run_dir/"
done
cp "$source_input" "$new_run_dir/input.in.upstream"
cp "$source_input" "$new_run_dir/input.in"
cp "$preflight" "$new_run_dir/guide-preflight.py"
printf 'Prepared: %s\nReview input.in before launching.\n' "$new_run_dir"
