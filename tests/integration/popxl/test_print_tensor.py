# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops


def test_print_tensor_format(capfd):
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = popxl.variable([0.0, 1.0], name="x")
        y = ops.print_tensor(x, "x", print_gradient=True, open_bracket="|")

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="out_stream")
        ops.host_store(y_d2h, y)

    capfd.readouterr()

    with popxl.Session(ir, "ipu_model") as session:
        session.run()

    captured = capfd.readouterr()
    stdoutput = captured.err

    assert "|" in stdoutput
