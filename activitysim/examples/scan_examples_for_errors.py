# This script looks for errors in the examples creates by `create_run_all_examples.py`

import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "working_dir",
    type=str,
    metavar="PATH",
    help="path to examples working directory",
)

args = parser.parse_args()

files_with_errors = []

for logfile in glob.glob(f"{args.working_dir}/*/output/log/activitysim.log"):
    with open(logfile, "rt") as f:
        printing_traceback = False
        found_traceback = False
        for n, line in enumerate(f.readlines(), start=1):
            if printing_traceback:
                print(line.rstrip())
                if not line.startswith(" "):
                    printing_traceback = False
            else:
                if "Traceback" in line:
                    print(f"======= TRACEBACK in {logfile} at line {n} =======")
                    print(line.rstrip())
                    printing_traceback = True
                    found_traceback = True
        if not found_traceback:
            print(f"OK: {logfile}")
        else:
            files_with_errors.append(logfile)

if files_with_errors:
    print("=====================================================")
    print(f"Found {len(files_with_errors)} examples with errors:")
    for f in files_with_errors:
        print(f"- {f}")
    print("=====================================================")
