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
  PYBIN=python2
fi

# Use the virtualenv for building
VE="${PWD}/../external/willow_build_python_${PYBIN}"
source ${VE}/bin/activate

cd build
./test.sh willow -VV

echo "Done"

