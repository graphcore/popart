#!/usr/bin/env bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
PROJECT_SOURCE_DIR=$1
CMAKE_CURRENT_BINARY_DIR=$2
YEAR=$(date +"%Y")
OUT_FILE="$CMAKE_CURRENT_BINARY_DIR/all_headers.hpp"
# customopbinder.hpp is excluded as it is compiled by cppimport, so the includes will not be found
# given it's directory setup. Instead, includes are specified by cppimport inside the
# file. The C++ 11 checker doesn't know this and flags this as an error.
declare -a EXCLUDES=("willow/include/popart/op/custom/parameterizedopbinder.hpp") # add further excludes here.

echo "// Copyright (c) $YEAR Graphcore Ltd. All rights reserved." > $OUT_FILE
for header in $(find $PROJECT_SOURCE_DIR/willow/include -name '*hpp' | sort);
  do
  for exclude in ${EXCLUDES[@]}; do
    if grep -q $exclude <<< "$header" ; then
      continue 2
    fi
  done
  printf "#include <$header>\n";
done >> $OUT_FILE
