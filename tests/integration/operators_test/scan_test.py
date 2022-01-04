# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("rev_d0", [False, True])
@pytest.mark.parametrize("rev_d1", [False, True])
@pytest.mark.parametrize("rev_out", [False, True])
@pytest.mark.parametrize("op", ["zip", "add", "mul"])
def test_scan_zip(op_tester, rev_d0, rev_d1, rev_out, op):
    d0 = np.array([0, 2, 4, 6]).astype(np.float32)
    d1 = np.array([1, 3, 5, 7]).astype(np.float32)

    def init_builder(builder):
        td0 = builder.addInputTensor(d0)
        td1 = builder.addInputTensor(d1)

        builder.setGraphName("main_graph")
        scan_builder = builder.createSubgraphBuilder()
        scan_builder.setGraphName("body")

        td0_in = scan_builder.addUntypedInputTensor(td0)
        td1_in = scan_builder.addUntypedInputTensor(td1)
        if op == "zip":
            out = scan_builder.aiOnnx.concat([td0_in, td1_in], 0)
        elif op == "add":
            out = scan_builder.aiOnnx.add([td0_in, td1_in])
        elif op == "mul":
            out = scan_builder.aiOnnx.mul([td0_in, td1_in])
        else:
            raise Exception(f'Unknown operation {op}')

        scan_builder.addOutputTensor(out)

        o = builder.aiOnnx.scan(
            [td0, td1],
            num_scan_inputs=2,
            body=scan_builder,
            scan_input_directions=[1 if rev_d0 else 0, 1 if rev_d1 else 0],
            scan_output_directions=[1 if rev_out else 0],
            num_outputs=1)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):

        d0r = d0[::(-1 if rev_d0 else 1)]
        d1r = d1[::(-1 if rev_d1 else 1)]

        if op == "zip":
            out = list(zip(d0r, d1r))
        elif op == "add":
            out = d0r + d1r
        elif op == "mul":
            out = d0r * d1r
        else:
            raise Exception(f'Unknown operation {op}')

        ref = (np.asarray(out)[::(-1 if rev_out else 1)]).flatten()
        return [ref]

    op_tester.run(init_builder, reference, step_type='infer')


@pytest.mark.parametrize("rev_d", [False, True])
@pytest.mark.parametrize("rev_out", [False, True])
def test_scan_basic_rnn(op_tester, rev_d, rev_out):
    np.random.seed(1302)
    data = np.random.randint(-4, 4, [8, 16, 32]).astype(np.float32)
    state = np.random.randint(-4, 4, [8, 24, 1]).astype(np.float32)
    w = np.random.randint(-4, 4, [1, 24, 32]).astype(np.float32)
    u = np.random.randint(-4, 4, [1, 24, 24]).astype(np.float32)
    v = np.random.randint(-4, 4, [1, 64, 24]).astype(np.float32)

    def init_builder(builder):
        td = builder.addInputTensor(data)
        ts = builder.addInputTensor(state)
        tw = builder.addInitializedInputTensor(w)
        tu = builder.addInitializedInputTensor(u)
        tv = builder.addInitializedInputTensor(v)

        builder.setGraphName("main_graph")
        scan_builder = builder.createSubgraphBuilder()
        scan_builder.setGraphName("body")

        ts_in = scan_builder.addUntypedInputTensor(ts)
        td_in = scan_builder.addUntypedInputTensor(td)

        td_in = scan_builder.reshape_const(scan_builder.aiOnnx, [td_in],
                                           [8, 32, 1])

        tuh = scan_builder.aiOnnx.matmul([tu, ts_in])
        twx = scan_builder.aiOnnx.matmul([tw, td_in])
        th = scan_builder.aiOnnx.add([tuh, twx])
        th = scan_builder.aiOnnx.tanh([th])
        ts_out = th
        td_out = scan_builder.aiOnnx.matmul([tv, th])

        scan_builder.addOutputTensor(ts_out)
        scan_builder.addOutputTensor(td_out)

        ots, otd = builder.aiOnnx.scan(
            [ts, td],
            num_scan_inputs=1,
            body=scan_builder,
            scan_input_axes=[1],
            scan_input_directions=[1 if rev_d else 0],
            scan_output_axes=[1],
            scan_output_directions=[1 if rev_out else 0],
            num_outputs=2)
        builder.addOutputTensor(ots)
        builder.addOutputTensor(otd)
        return [ots, otd]

    def reference(ref_data):
        class BasicRNNModule(torch.nn.Module):
            def __init__(self):
                super(BasicRNNModule, self).__init__()

            def forward(self, ts_in, td_in, tw_in, tu_in, tv_in, axis_in,
                        axis_out, rev_in, rev_out):
                ts_out = ts_in
                td_out = []
                for i in range(td_in.shape[axis_in]):
                    tuh = torch.matmul(tu, ts_in)
                    td_in_s = torch.split(
                        td_in, 1, dim=axis_in)[::(-1 if rev_in else 1)][i]
                    td_in_s = torch.reshape(td_in_s, [8, 32, 1])
                    twx = torch.matmul(tw, td_in_s)
                    th = tuh + twx
                    th = torch.tanh(th)
                    ts_out = th
                    td_out += [torch.matmul(tv, th)]
                    ts_in = ts_out
                return ts_out, torch.cat(td_out[::(-1 if rev_out else 1)],
                                         dim=axis_out)

        td = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        ts = torch.tensor(state, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        tu = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        tv = torch.tensor(v, dtype=torch.float32, requires_grad=True)
        model = BasicRNNModule()
        ots, otd = model(ts, td, tw, tu, tv, 1, 1, rev_d, rev_out)
        return [ots, otd]

    op_tester.rtol = 1e-03
    op_tester.atol = 1e-04
    op_tester.run(init_builder, reference, step_type='infer')
