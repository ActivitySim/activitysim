#!/usr/bin/env bash

set -xe

env | sort

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo "***** Start: Building Activitysim installer *****"
CONSTRUCT_ROOT="${CONSTRUCT_ROOT:-${PWD}}"

cd "${CONSTRUCT_ROOT}"

# Constructor should be latest for non-native building
# See https://github.com/conda/constructor
echo "***** Install constructor *****"
conda install -y "constructor>=3.3.1" jinja2 curl libarchive -c conda-forge --override-channels

if [[ "$(uname)" == "Darwin" ]]; then
    conda install -y coreutils -c conda-forge --override-channels
fi
# shellcheck disable=SC2154
if [[ "${TARGET_PLATFORM}" == win-* ]]; then
    conda install -y "nsis>=3.01" -c conda-forge --override-channels
fi
# pip install git+git://github.com/conda/constructor@3.3.1#egg=constructor --force --no-deps
conda list

echo "***** Make temp directory *****"
TEMP_DIR=$(mktemp -d --tmpdir=C:/Users/RUNNER~1/AppData/Local/Temp/);

echo "***** Copy file for installer construction *****"
cp -R other_resources/installer "${TEMP_DIR}/"
cp LICENSE.txt "${TEMP_DIR}/installer/"

ls -al "${TEMP_DIR}"

echo "***** Construct the installer *****"
# Transmutation requires the current directory is writable
cd "${TEMP_DIR}"
# shellcheck disable=SC2086
constructor "${TEMP_DIR}/installer/" --output-dir "${TEMP_DIR}"
cd -

cd "${TEMP_DIR}"

# This line will break if there is more than one installer in the folder.
INSTALLER_PATH=$(find . -name "Activitysim*.${EXT}" | head -n 1)

echo "***** Move installer to build folder *****"
mkdir -p "${CONSTRUCT_ROOT}/build"
mv "${INSTALLER_PATH}" "${CONSTRUCT_ROOT}/build/"

echo "***** Done: Building ActivitySim installer *****"
cd "${CONSTRUCT_ROOT}"
