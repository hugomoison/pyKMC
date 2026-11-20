#!/bin/bash
# Copy all Si inputs from the installed checkout into a fresh working directory.
set -e
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash prepare_example.sh /path/to/pykmc_install /path/to/new-run\n' >&2
    exit 2
fi
source_example="$1/pyKMC/examples/Si_vac"
new_run_dir="$2"
if [[ -s "$source_example/input.in.template" ]]; then
    source_input="$source_example/input.in.template"
elif [[ -s "$source_example/input.in" ]]; then
    source_input="$source_example/input.in"
else
    printf 'No input.in.template or input.in found in %s\n' "$source_example" >&2
    exit 2
fi
for name in initial_config.xyz Si.sw; do
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
cp "$source_example/initial_config.xyz" "$source_example/Si.sw" "$new_run_dir/"
cp "$source_input" "$new_run_dir/input.in.upstream"
cp "$source_input" "$new_run_dir/input.in"
cp "$preflight" "$new_run_dir/guide-preflight.py"
printf 'Prepared: %s\n' "$new_run_dir"
printf 'Review input.in using the Si README: correct initial_config and remove ignored legacy options if present.\n'
printf 'Keep the supplied physical/search settings for the baseline. No simulation was started.\n'
