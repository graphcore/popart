#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
USAGE_MSG="USAGE: $(basename "$0") [INSTALL-DIR] [CLANG-LIB-DIR] [VERSION]"

if [ "$#" -ne 3 ]; then
    printf '%s\n' "$USAGE_MSG"
    exit
fi

INSTALL_DIR="$1"
CLANG_LIB_DIR="$2"
VERSION="$3"

INSTALL_FILES_DIR="$INSTALL_DIR/../lint_install_files"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_FILES_DIR"
cd "$INSTALL_FILES_DIR" || exit  # cd one level above the install directory

git clone --depth 1 --branch v"$VERSION" https://github.com/oclint/oclint.git
cd oclint/oclint-scripts || exit
./makeWithSystemLLVM "$CLANG_LIB_DIR"
cd ../build/oclint-release || exit
cp bin/oclint* "$INSTALL_DIR/bin/"
cp -rp lib/* "$INSTALL_DIR/lib/"

# Some versions of oclint includes an include directory as well
if [ -d include ]; then
    cp -rp include/* "$INSTALL_DIR/include/"
fi
