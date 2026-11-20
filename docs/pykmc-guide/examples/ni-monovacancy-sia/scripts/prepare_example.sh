#!/bin/bash
# Copy all three inputs from the installed vacancy+SIA example, without edits.
set -e
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash prepare_example.sh /path/to/pykmc_install /path/to/new-run\n' >&2
    exit 2
fi
source_example="$1/pyKMC/examples/Ni_fcc_4000at_monovacancy+sia"
new_run_dir="$2"
if [[ -s "$source_example/input.in.template" ]]; then
    source_input="$source_example/input.in.template"
elif [[ -s "$source_example/input.in" ]]; then
    source_input="$source_example/input.in"
else
    printf 'No input.in.template or input.in found in %s\n' "$source_example" >&2
    exit 2
fi
for name in initial_config.xyz Ni_v6_2.0_LKBeland2016.eam; do
    if [[ ! -s "$source_example/$name" ]]; then
        printf 'Missing input: %s/%s\n' "$source_example" "$name" >&2
        exit 2
    fi
done
if [[ -e "$new_run_dir" || -L "$new_run_dir" ]]; then
    printf 'Destination already exists; nothing was changed: %s\n' "$new_run_dir" >&2
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
cp "$source_example/initial_config.xyz" "$source_example/Ni_v6_2.0_LKBeland2016.eam" "$new_run_dir/"
cp "$source_input" "$new_run_dir/input.in.upstream"
cp "$source_input" "$new_run_dir/input.in"
cp "$preflight" "$new_run_dir/guide-preflight.py"
printf 'Prepared: %s\n' "$new_run_dir"
printf 'Before running, edit input.in: initial_config = ./initial_config.xyz and n_steps = 300.\n'
printf 'Review the session layout and remaining settings in the guide. No simulation was started.\n'
