#!/bin/bash
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
set -e
export PYTHONPATH=./python:popart/python
python -s $1
