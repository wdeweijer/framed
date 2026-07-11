#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
crate_dir="$(cd -- "$script_dir/.." && pwd)"
visualizations_dir="${1:-"$crate_dir/visualizations"}"

if [[ ! -d "$visualizations_dir" ]]; then
    echo "visualizations directory not found: $visualizations_dir" >&2
    exit 1
fi

shopt -s nullglob
dot_files=("$visualizations_dir"/*.dot)

if (( ${#dot_files[@]} == 0 )); then
    echo "no .dot files found in $visualizations_dir"
    exit 0
fi

for dot_file in "${dot_files[@]}"; do
    svg_file="${dot_file%.dot}.svg"

    if [[ -e "$svg_file" && ! "$dot_file" -nt "$svg_file" ]]; then
        echo "skipped $svg_file"
        continue
    fi

    if grep -q 'layout=neato' "$dot_file"; then
        neato -n2 -Tsvg "$dot_file" -o "$svg_file"
    else
        dot -Tsvg "$dot_file" -o "$svg_file"
    fi

    echo "wrote $svg_file"
done
