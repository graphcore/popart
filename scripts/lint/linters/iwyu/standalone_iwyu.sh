#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# This script calls include_what_you_use_linter.sh with all the
# repository files, excluding the excluded_files below.

excluded_files=(
    popart/opidentifier.hpp
    verify_cxx_11_interface.cpp
    include/popart/vendored/
)
# https://stackoverflow.com/a/9429887/8791653
excluded_files_pattern=$(IFS='|' ; echo "${excluded_files[*]}")

find . -type f -name \*\.cpp -o -name \*\.hpp | \
    grep -v -P "$excluded_files_pattern" | \
    scripts/lint/linters/iwyu/include_what_you_use_linter.sh
