#!/bin/bash -e
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash popart/ci/test.sh' from the poponnx_view directory."
  exit 1
fi

source ./popart/ci/utils.sh

if [ $# -gt 0 ]
then
  PYBIN=$1
else
  PYBIN=python3
fi

# Use the virtualenv for building
VE="${PWD}/../external/popart_build_python_${PYBIN}"
source ${VE}/bin/activate

cd build
./test.sh popart -VV

cd build/popart
make popart_run_examples

echo "Done"

