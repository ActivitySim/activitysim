#!/bin/bash

# run the primary example
# source ../../opt/conda/etc/profile.d/conda.sh
# conda activate ASIM-DEV
# activitysim create -e example_mtc -d test_example_mtc
# cd test_example_mtc
# activitysim run -c configs -o output -d data

# run the summarize example
source ../../opt/conda/etc/profile.d/conda.sh
conda activate ASIM-DEV
pytest activitysim/abm/test/test_misc/test_summarize.py