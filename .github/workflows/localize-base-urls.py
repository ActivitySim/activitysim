import argparse
import os
import pathlib

GITHUB_REPOSITORY_OWNER = os.environ.get(
    "GITHUB_REPOSITORY_OWNER", "ActivitySim"
).lower()

parser = argparse.ArgumentParser()
parser.add_argument("path", type=pathlib.Path)
args = parser.parse_args()

if GITHUB_REPOSITORY_OWNER != "activitysim":
    content = args.path.read_text()
    new_content = content.replace(
        "activitysim.github.io", f"{GITHUB_REPOSITORY_OWNER}.github.io"
    )
    args.path.write_text(new_content)
