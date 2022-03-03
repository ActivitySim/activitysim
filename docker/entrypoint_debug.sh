#!/bin/bash

# run tests in live debug
source ../../opt/conda/etc/profile.d/conda.sh
conda activate ASIM-DEV
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ../../opt/conda/envs/ASIM-DEV/bin/pytest activitysim/abm/test/test_misc/test_summarize.py