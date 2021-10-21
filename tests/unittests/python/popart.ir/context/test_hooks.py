# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List, Type
import numpy as np
import pytest

import popart._internal.ir as _ir
import popart.ir as pir


def test_hook():
    ir = pir.Ir()
    g = ir.main_graph()

    called = False

    def hook(_):
        nonlocal called
        called = True

    handle = g.register_op_created_hook(hook)

    with g:
        x = pir.variable(1)
        x = x + 1

    assert called
    called = False

    # Creating this graph will create
    # an AddOp on the new graph.
    # Ensure this does not trigger the hook.
    sg = ir.create_graph(lambda y: y + 1, x)
    assert not called

    g.remove_op_created_hook(handle)
    with g:
        x = x + 1
    assert not called


def test_multiple_hooks():
    ir = pir.Ir()
    g = ir.main_graph()

    called_1 = False
    called_2 = False

    def hook_1(_):
        nonlocal called_1
        called_1 = True

    def hook_2(_):
        nonlocal called_2
        called_2 = True

    g.register_op_created_hook(hook_1)
    g.register_op_created_hook(hook_2)

    with g:
        x = pir.variable(1)
        x = x + 1

    assert called_1 and called_2
