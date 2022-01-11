#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
USAGE_MSG="USAGE: $(basename "$0") [INSTALL-DIR] [CLANG-LIB-DIR] [CLANG-VERSION]"

if [ "$#" -ne 3 ]; then
    printf '%s\n' "$USAGE_MSG"
    exit
fi

INSTALL_DIR="$1"
CLANG_LIB_DIR="$2"
CLANG_VERSION="$3"

INSTALL_FILES_DIR="$INSTALL_DIR/../lint_install_files"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_FILES_DIR"
cd "$INSTALL_FILES_DIR" || exit  # cd one level above the install directory

git clone --depth 1 --branch clang_"$CLANG_VERSION" https://github.com/include-what-you-use/include-what-you-use
cd include-what-you-use || exit
mkdir build
cd build || exit
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_PREFIX_PATH="$CLANG_LIB_DIR" -G "Ninja" ..
ninja install -j96
