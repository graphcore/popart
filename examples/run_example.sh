#!/bin/bash
set -e
export PYTHONPATH=./python:popart/python
python -s $1
