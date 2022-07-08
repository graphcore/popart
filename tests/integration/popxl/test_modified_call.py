# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops


def test_modified():
    ir = popxl.Ir()
    g = ir.main_graph

    with g, popxl.in_sequence():
        x = popxl.variable(1)

        sg = ir.create_graph(
            lambda x: ops.var_updates.accumulate_(x, popxl.constant(1)), x
        )

        ops.call(sg, x)
        # Store x
        x_non_modify_stream = popxl.d2h_stream(x.shape, x.dtype)
        ops.host_store(x_non_modify_stream, x)

        info = ops.call_with_info(sg, x)
        info.set_parent_input_modified(x)
        x_modifiy_stream = popxl.d2h_stream(x.shape, x.dtype)
        ops.host_store(x_modifiy_stream, x)

    session = popxl.Session(ir, "ipu_model")

    with session:
        outputs = session.run()

    assert outputs[x_non_modify_stream] == 1
    assert outputs[x_modifiy_stream] == 2
