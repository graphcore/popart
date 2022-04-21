#!/usr/bin/env bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

if [ ! "$POPLAR_VIEW_BUILD_DIR" ]; then
    echo "IWYU (include-what-you-use) failed to access the build directory"
    echo "The include_what_you_use hook needs to know where the poplar view build directory is located to work properly."
    echo "You can provide this information by setting the \$POPLAR_VIEW_BUILD_DIR environment variable."
    echo "Set it up by putting the following in your .bashrc: export POPLAR_VIEW_BUILD_DIR=<view_build_dir>"
    echo "An alternative is to set the environment variable for the current command only. For example POPLAR_VIEW_BUILD_DIR=<temporary_view_build_dir> <command_which_triggers_this_script>"
    echo "You can also disable IWYU by specifying the environment variable SKIP=iwyu"
    exit 1
fi
if [ ! -d "$POPLAR_VIEW_BUILD_DIR"/install/poplar/include ]; then
    echo "IWYU (include-what-you-use) failed to access the build directory"
    echo "\$POPLAR_VIEW_BUILD_DIR is set to '$POPLAR_VIEW_BUILD_DIR' which is not a valid build directory."
    echo "It's either not a valid direcory or you haven't built the view with 'ninja popart' yet"
    echo "You can also disable IWYU by specifying the environment variable SKIP=iwyu"
    exit 1
fi
# https://stackoverflow.com/questions/35071192/how-to-find-out-where-the-python-include-directory-is
python_path=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")

# Note that include-what-you-use only takes 1 file at a time, hence we need to use xargs
echo $* | xargs -n 1 \
    include-what-you-use \
    -Xiwyu \
    --mapping_file=scripts/lint/linters/iwyu/all_mappings.imp \
    -Xiwyu \
    --max_line_length=200 \
    -Iwillow/include \
    -Iwillow/src \
    -Itests/unittests \
    -Itests/testutil/include \
    -Ipython/popart._internal.ir \
    -I"$python_path" \
    -I"$POPLAR_VIEW_BUILD_DIR"/build/popart/willow/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/spdlog/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/capnproto/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/onnx/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/boost/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/protobuf/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/popef/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/poplibs/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/poplar/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/poprithms/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/libpvti/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/libpva/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/gcl/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/popir/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/gccs/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/pybind11/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/boost/include \
    -isystem "$POPLAR_VIEW_BUILD_DIR"/install/trompeloeil/include/ \
    -DONNX_NAMESPACE=onnx \
    -w

exit $?
