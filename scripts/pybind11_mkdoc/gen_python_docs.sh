#!/bin/bash -e
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Run this file with either a poplar SDK include directory or the
# poplar_view build directory:
# bash gen_python_docs.sh path/to/poplar-install/include
# This will generate willow/include/popart/pydocs_popart_core.hpp which contains the docstrings
# for the python bindings. See the github repo above for details on how to access them, or
# look at some other docstrings for examples.

# If run from inside Poplarview, then LLVM and LIBCLANG directories are identified automatically.
# Otherwise they must be specified by the user using LLVM_DIR_PATH and LIBCLANG_PATH environment parameters.

# Re-run this every time you make a change to .hpp dcumentation you want reflected in the
# python docs. E.g. if you update the doxygen comments to sessionoptions.globalReplicationFactor
# in willow/include/popart/sessionoptions.hpp, Run this, and it will copy the changes to pydocs_popart_core.hpp
# and the changes will appear in the python docs.

# Check a poplar include or build dir was supplied.
if [ -z "$1" ]
then
    echo "Please supply a poplar include directory or poplar_view build dir:"
    echo $0 "<poplar include directory or build dir>"
    exit 0
else
    # Check for poplar_view
    cwd=$(pwd)
    cd "$(readlink -f $1)"
    if [[  $(git config --get remote.origin.url) == *"POPLARVIEW"* ]]
    then
        echo "In poplar_view, assuming a build dir"
        EXTRA_INCLUDES_STR="-I $1/install/gccs/include \
        -I $1/install/gcl/include \
        -I $1/install/libpva/include \
        -I $1/install/popir/include \
        -I $1/install/poplar/include \
        -I $1/install/poplibs/include \
        -I $1/install/poprithms/include"
        # If LLVM_DIR_PATH is not set, give it a default value inside the POPLARVIEW build directory
        # LIBCLANG_PATH in this case should be automatically found inside $LLVM_DIR_PATH
        if [ -z "$LLVM_DIR_PATH" ]
        then
            LLVM_DIR_PATH="$1/install/llvm"
        fi
    else
        echo "Not in poplar_view, assuming a poplar SDK include dir"
        EXTRA_INCLUDES_STR="-I $1"

        if [[ -z "$LLVM_DIR_PATH" || -z "$LIBCLANG_PATH" ]]
        then
            echo "Please specify llvm directory and libclang.so paths through LLVM_DIR_PATH and LIBCLANG_PATH environment variables when running outside of POPLARVIEW"
            exit 0
        fi
    fi
    cd "$cwd"
fi

# Run the generation command.
# The below hpp files are taken from python/popart.cpp and seem to be sufficient to give
# pybind11_mkdoc enough info to generate the required docstrings. In future, more hpp files may need to
# be added to ensure full coverage, mostly decided by trial and error.
# TODO: T29154 generate this list of files automatically.

echo "Generating python docs from .hpp files:"

OUT_FILE="willow/include/popart/docs/pydocs_popart_core.hpp"
CMD="PYTHONPATH=./scripts:$PYTHONPATH \
LLVM_DIR_PATH=$LLVM_DIR_PATH \
CPATH=$CPATH:/usr/include/c++/4.8.5/x86_64-redhat-linux/ \
python3 -m pybind11_mkdoc \
-I /usr/include \
$EXTRA_INCLUDES_STR \
-I willow/include \
-ferror-limit=100000 \
-DONNX_NAMESPACE=onnx \
willow/include/popart/adam.hpp \
willow/include/popart/builder.hpp \
willow/include/popart/clipnormsettings.hpp \
willow/include/popart/commgroup.hpp \
willow/include/popart/devicemanager.hpp \
willow/include/popart/dataflow.hpp \
willow/include/popart/op.hpp \
willow/include/popart/error.hpp \
willow/include/popart/graphtransformer.hpp \
willow/include/popart/ir.hpp \
willow/include/popart/numerics.hpp \
willow/include/popart/op/collectives/collectives.hpp \
willow/include/popart/op/exchange/exchange.hpp \
willow/include/popart/op/identity.hpp \
willow/include/popart/op/init.hpp \
willow/include/popart/op/l1.hpp \
willow/include/popart/op/nll.hpp \
willow/include/popart/opmanager.hpp \
willow/include/popart/optimizer.hpp \
willow/include/popart/optimizervalue.hpp \
willow/include/popart/patterns/patterns.hpp \
willow/include/popart/popx/devicex.hpp \
willow/include/popart/replicatedstreammode.hpp \
willow/include/popart/session.hpp \
willow/include/popart/sessionoptions.hpp \
willow/include/popart/sgd.hpp \
willow/include/popart/stepio_generic.hpp \
willow/include/popart/stepio_size_assertion.hpp \
willow/include/popart/tensordata.hpp \
willow/include/popart/tensorlocation.hpp \
willow/include/popart/tensornames.hpp \
willow/include/popart/tensors.hpp \
willow/include/popart/tensor.hpp \
willow/include/popart/variablesettings.hpp \
-o $OUT_FILE"

eval $CMD

arc lint "$OUT_FILE"
