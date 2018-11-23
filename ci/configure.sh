#!/bin/bash -e

set -e

if [ ! -f "view.txt" ]
then
  echo "Run 'bash willow/ci/configure.sh' from the willow_view directory."
  exit 1
fi

source ./willow/ci/utils.sh

if [ $# -gt 0 ]
then
  PYBIN=$1
else
  PYBIN=python3
fi

case ${PYBIN} in
python2)
  PYPKG="python@2"
  ;;
python3)
  PYPKG="python@3"
  ;;
*)
  echo "configure [python2|python3]"
  ;;
esac

# Find the right executable on an OS/X homebrew platform
if [ -x "$(command -v brew)" ]
then
  PYTHON_BIN_PATH=`brew list ${PYPKG} | grep "python$" | head -n 1`
fi

if [ -z ${PYTHON_BIN_PATH} ]
then
  PYTHON_BIN_PATH=`which ${PYBIN}`
fi

echo "Using ${PYTHON_BIN_PATH}"

VE="${PWD}/../external/willow_build_python_${PYBIN}"

# Set up an independent python virtualenv
rm -rf ${VE}
virtualenv -p ${PYTHON_BIN_PATH} ${VE}
source ${VE}/bin/activate

# Install dependencies
pip install numpy
pip install pytest
pip install yapf
pip install torchvision

# Create a directory for building
rm -rf build
mkdir build
cd build

# Create the superpiroject
../cbt/cbt.py ..

# Number of processors
NUM_PROCS=$(get_processor_count)

# Configure cmake
CC=clang CXX=clang++   cmake . -DPOPLAR_INSTALL_DIR=`readlink -f ../../external/poplar-install/` -DEXTERNAL_PROJECT_NUM_JOBS=${NUM_PROCS} -DPOPONNX_CMAKE_ARGS=-DBUILD_DOCS=ON

echo "Done"

