#!/bin/bash -e

# How to use:
# First, install the mkdoc python module:
# `python3 -m pip install git+git://github.com/pybind/pybind11_mkdoc.git@master`
# Also install python-clang if you don't have it already:
# `pip install clang`
# Then run this file with a poplar include directory :
# bash gen_python_docs.sh path/to/poplar-install/include
# This will generate willow/include/popart/pydocs_popart_core.hpp which contains the docstrings
# for the python bindings. See the githib repo above for details on how to access them, or
# look at some other docstrings for examples.

# Re-run this every time you make a change to .hpp dcumentation you want reflected in the
# python docs. E.g. if you update the doxygen comments to sessionoptions.globalReplicationFactor
#  in willow/include/popart/sessionoptions.hpp, Run this, and it will copy the changes to pydocs_popart_core.hpp
# and the changes will appear in the python docs.

CMD="python3 -c 'import pkgutil; print(1 if pkgutil.find_loader(\"pybind11_mkdoc\") else 0)'"
INSTALLED=`eval $CMD`

# Check python module installed
if [[ $INSTALLED -eq 0 ]]
then
    echo "Python pybind11_mkdoc package not found."
    echo "Please install from:"
    echo "https://github.com/pybind/pybind11_mkdoc"
    exit 0
fi

# Check poplar include dir was supplied.
if [ -z "$1" ]
then
    echo "Please supply a poplar include directory:"
    echo $0 "<poplar include directory>"
    exit 0
fi

# Run the generation command.
# The below hpp files are taken from python/popart.cpp and seem to be sufficient to give
# pybind11_mkdoc enough info to generate the required docstrings. In future, more hpp files may need to
# be added to ensure full coverage, mostly decided by trial and error.
# TODO: T29154 generate this list of files automatically.

echo "Generating python docs from .hpp files:"
CMD="python3 -m pybind11_mkdoc \
-I /usr/include \
-I "$1" \
-I willow/include \
-ferror-limit=100000 \
-DONNX_NAMESPACE=onnx \
willow/include/popart/adam.hpp \
willow/include/popart/builder.hpp \
willow/include/popart/devicemanager.hpp \
willow/include/popart/dataflow.hpp \
willow/include/popart/op.hpp \
willow/include/popart/error.hpp \
willow/include/popart/graphtransformer.hpp \
willow/include/popart/ir.hpp \
willow/include/popart/numerics.hpp \
willow/include/popart/op/collectives/collectives.hpp \
willow/include/popart/op/identity.hpp \
willow/include/popart/op/init.hpp \
willow/include/popart/op/l1.hpp \
willow/include/popart/op/nll.hpp \
willow/include/popart/opmanager.hpp \
willow/include/popart/optimizer.hpp \
willow/include/popart/optimizervalue.hpp \
willow/include/popart/patterns/patterns.hpp \
willow/include/popart/popx/devicex.hpp \
willow/include/popart/session.hpp \
willow/include/popart/sessionoptions.hpp \
willow/include/popart/stepio_generic.hpp \
willow/include/popart/stepio_size_assertion.hpp \
willow/include/popart/tensordata.hpp \
willow/include/popart/tensorlocation.hpp \
willow/include/popart/tensornames.hpp \
willow/include/popart/tensors.hpp \
willow/include/popart/tensor.hpp \
-o willow/include/popart/docs/pydocs_popart_core.hpp"

eval $CMD
