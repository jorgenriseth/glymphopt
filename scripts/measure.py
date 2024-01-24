import argparse
from pathlib import Path

from glymphopt.measure import T1_measurement_from_file, measure_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=Path, required=True)
parser.add_argument("--resolution", type=int, required=True)
parser.add_argument("--timestamps_out", type=Path, required=True)
parser.add_argument("--measure_T1", action="store_true")

args = parser.parse_args()
if args.measure_T1:
    T10 = {1: 1.0, 2: 1.2}
    T1_measurement_from_file(
        args.infile, "concentration", args.resolution, T10, args.timestamps_out
    )
else:
    measure_from_file(
        args.infile, "concentration", args.resolution, args.timestamps_out
    )
