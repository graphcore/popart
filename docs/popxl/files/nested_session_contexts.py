# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to demonstrate the semantics of nested session
contexts.
'''

import popxl
import popxl.ops as ops
import numpy as np

# Creating a model with PopXL
ir = popxl.Ir()
main = ir.main_graph

with main:
    a = popxl.variable(3, dtype=popxl.int32, name="variable_a")
    b = popxl.constant(1, dtype=popxl.int32, name="constant_b")

    # addition
    o = a + b
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

# Construct an Ir `ir`...

ir.num_host_transfers = 1

# Session begin

# Enter Ctxt 1, causes attach, weights_from_host
with popxl.Session(ir, "ipu_model") as session:
    # Attach, weights_from_host
    assert session.device.isAttached

    # Enter Ctxt 2, still attached, no weights_from_host again
    with session:
        assert session.device.isAttached

        # Manual detach then enter Ctxt 3. Causes attach and weights_from_host
        session.device.detach()
        with session:
            assert session.device.isAttached
        # Exit Ctxt 3, causes detach and weights_to_host, as detached on enter
        assert not session.device.isAttached

        # Manual re-attach
        session.device.attach()

    # Exit Ctxt 2, no detach or weights_to_host, as attached on enter
    assert session.device.isAttached

# Exit Ctxt3, causes detach and weights weights_to_host as detached on enter
assert not session.device.isAttached

# Session end
