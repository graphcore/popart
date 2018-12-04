#!/bin/bash -e

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash poponnx/ci/upload.sh' from the poponnx_view directory."
  exit 1
fi

source ./poponnx/ci/utils.sh

if [ $# -eq 0 ]
then
  echo "upload.sh <swdb url> [python2|python3]"
  exit 1
fi

if [ $# -gt 1 ]
then
  PYBIN=$2
else
  PYBIN=python3
fi

# Use the virtualenv for building
VE="${PWD}/../external/poponnx_build_python_${PYBIN}"
source ${VE}/bin/activate

rm -rf build/pkg

pushd build/build/poponnx
make package_and_move
popd

echo "== Package contents =="
ls -1 build/pkg
echo "======================"

swdb_api/swdb_upload.py $1 build/pkg

echo "Done"

