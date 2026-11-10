#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: bash $0 INPUT_HDF OUTPUT_DIR [A ...]" >&2
    exit 1
fi

input_hdf=$(realpath -- "$1")
mkdir -p -- "$2"
output_dir=$(cd -- "$2" && pwd)
shift 2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."

a_values=("$@")
if (( ${#a_values[@]} == 0 )); then
    a_values=(1 2 3 5)
fi
r=1e-5
dt=3600
data_csv="$output_dir/measured_amount.csv"
data_args=(--data-output "$data_csv")
plot_args=()

for a in "${a_values[@]}"; do
    csv="$output_dir/amount_a-${a}_r-${r}_dt-${dt}.csv"
    pixi run python "$script_dir/singlecompartment_amount.py" \
        -i "$input_hdf" -o "$csv" "${data_args[@]}" \
        --a "$a" --r "$r" --dt "$dt"
    data_args=()
    plot_args+=(-i "$csv" --label "a=$a")
done

pixi run python "$script_dir/plot_amount_mesh.py" \
    "${plot_args[@]}" --data "$data_csv" -o "$output_dir/amount_comparison.png"
