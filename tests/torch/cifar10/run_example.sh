#!/bin/bash
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.

THIS_DIR=`dirname $0`

export PYTHONPATH=${THIS_DIR}/../../../python
export LD_LIBRARY_PATH=${THIS_DIR}/../../../lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${THIS_DIR}/../../../lib:${DYLD_LIBRARY_PATH}

MODEL=${THIS_DIR}/$1
shift

python ${MODEL} $@

