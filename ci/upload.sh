#!/bin/bash -e

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash willow/ci/test.sh' from the willow_view directory."
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

cd build/build/willow
make package_and_move

echo "Done"

