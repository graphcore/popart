#!/bin/bash -e
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.

set -e

# Should be run with the poponnx_view directory as the CWD.
#
# This downloads the snapshot of poplar found in tbe config
# file, and unpacks it into ../external/poplar-install

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ] ; then
  echo "download_poplar_release.sh <swdb_url> <element type> [<source_id>]"
  exit 1
fi

if [ ! -d popart ]; then
  echo "CWD needs to be poponnx_view"
  exit 1
fi

source ./popart/ci/utils.sh

# Get the current directory
VIEW_DIR=${PWD}

# Put release in the 'external' directory
rm -rf ../external
mkdir -p ../external
cd ../external

# The element name
if [ $(uname) == 'Linux' ] ; then
  ELEMENT_NAME='Poplar ubuntu 18 04 installer'
else
  ELEMENT_NAME='Poplar osx installer'
fi

if [ $# -eq 3 ]; then
   sourceid_filter=("--source_id" "$3")
else
   sourceid_filter=()
fi

# Download
python ${VIEW_DIR}/swdb_api/swdb_download_element.py \
       ${sourceid_filter[@]} \
       --product_name poplar \
       --element_name "${ELEMENT_NAME}" \
       --swdb_url $1 poplar_installer.tar.gz

# Extract
tar xvzf poplar_installer.tar.gz

# Remove existing install if necessary
rm -rf poplar-install

# Now rename to a well known name
find . -maxdepth 1 -type d -name 'poplar*' -exec mv {} poplar-install \;

echo "Done"


