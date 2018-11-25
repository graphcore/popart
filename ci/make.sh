#!/bin/bash -e

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash poponnx/ci/make.sh' from the poponnx_view directory."
  exit 1
fi

source ./poponnx/ci/utils.sh

if [ $# -gt 0 ]
then
  PYBIN=$1
else
  PYBIN=python3
fi

# Use the virtualenv for building
VE="${PWD}/../external/poponnx_build_python_${PYBIN}"
source ${VE}/bin/activate

# Number of processors
NUM_PROCS=$(get_processor_count)

# Now buid
cd build
make -j ${NUM_PROCS}

echo "Done"

