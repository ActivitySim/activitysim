#!/usr/bin/env bash

set -ex

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda install posix --yes
source other_resources/installer/build.sh
