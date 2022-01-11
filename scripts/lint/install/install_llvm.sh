#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
USAGE_MSG="USAGE: $(basename "$0") [INSTALL-DIR] [VERSION]"

if [ "$#" -ne 2 ]; then
    printf '%s\n' "$USAGE_MSG"
    exit
fi

INSTALL_DIR="$1"
VERSION="$2"

INSTALL_FILES_DIR="$INSTALL_DIR/../lint_install_files"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_FILES_DIR"
cd "$INSTALL_FILES_DIR" || exit  # cd one level above the install directory

git clone --depth 1 --branch llvmorg-"$VERSION" https://github.com/llvm/llvm-project.git
cd llvm-project || exit
mkdir build
cd build || exit
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -G "Ninja" ../llvm
ninja install -j96
