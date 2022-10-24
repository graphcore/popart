# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
from popxl import ops


def test_autodiff_hassideeffect_op(capfd):
    def module(x):
        # PrintTensor hasSideEffect == True when printSelf==True
        x = ops.print_tensor(x, print_self=True, print_gradient=True)
        x = x @ x
        return x

    ir = popxl.Ir()

    with ir.main_graph:
        x = popxl.variable(np.arange(4), popxl.float32, "x")

        graph = ir.create_graph(module, x)

        dgraph = popxl.transforms.autodiff(graph)

        fwd_call = ops.call_with_info(graph, x)
        y, *_ = fwd_call.outputs

        (dx,) = ops.call(dgraph.graph, y, inputs_dict=dgraph.inputs_dict(fwd_call))

        dx_d2h = popxl.d2h_stream(dx.shape, dx.dtype, name="dx_stream")
        ops.host_store(dx_d2h, dx)

    capfd.readouterr()

    with popxl.Session(ir, "ipu_model") as session:
        session.run()

    captured = capfd.readouterr()
    stdoutput = captured.err

    assert "print_x" in stdoutput
    assert "print_x_gradient" in stdoutput
