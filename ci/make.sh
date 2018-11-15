#!/bin/bash -e

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash willow/ci/make.sh' from the willow_view directory."
  exit 1
fi

source ./willow/ci/utils.sh

if [ $# -gt 0 ]
then
  PYBIN=$1
else
  PYBIN=python3
fi

# Use the virtualenv for building
VE="${PWD}/../external/willow_build_python_${PYBIN}"
source ${VE}/bin/activate

# Number of processors
NUM_PROCS=$(get_processor_count)

# Now buid
cd build
make -j ${NUM_PROCS}

echo "Done"

