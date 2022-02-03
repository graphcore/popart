# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np

import popart
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.dtypes as dtypes
import popart.ir.ops as ops
from popart.ir.remote_buffer import RemoteBuffer
from popart.ir.tensor import Tensor
from popart.ir.transforms.autodiff import ExpectedConnection, ExpectedConnectionType
# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import test_util as tu

BPS = 1
ACCL_FACTOR = 1
COMP_BATCH = 1
HIDDEN_SIZE = 4
NUM_LOCAL_REPLICAS = 4
REMOTE_BUFFER_ID = 1
REMOTE_BUFFER_IDX = 0

np.random.seed(0)
weight_data = np.random.rand(NUM_LOCAL_REPLICAS,
                             HIDDEN_SIZE).astype(np.float32)

input_data = []
label_data = []

for i in range(0, BPS * NUM_LOCAL_REPLICAS * ACCL_FACTOR * COMP_BATCH):
    np.random.seed(i * NUM_LOCAL_REPLICAS)
    input_data += [np.random.rand(HIDDEN_SIZE).astype(np.float32)]
    label_data += [np.random.randint(0, HIDDEN_SIZE, size=1)]

input_data: np.ndarray = np.concatenate(input_data)
label_data: np.ndarray = np.concatenate(label_data).astype(np.uint32)


class TestTensorLocation():
    class Linear(pir.Module):
        def __init__(self):
            self.W: Tensor = None

        def build(self, d0: Tensor) -> Tensor:
            self.W = pir.subgraph_input((4, 4), pir.float32, "w0")
            y = d0 @ self.W
            return y

    @tu.requires_ipu_model
    def test_rts(self):

        ir = pir.Ir()
        data: Mapping[str, np.ndarray] = {}
        main = ir.main_graph()
        opts = ir._pb_ir.getSessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = NUM_LOCAL_REPLICAS

        ### Forward pass ###
        with main:
            d0_h2d = pir.h2d_stream((COMP_BATCH, HIDDEN_SIZE),
                                    pir.float32,
                                    name="d0_stream")
            d0 = ops.host_load(d0_h2d, "d")

            data[d0_h2d.tensor_id()] = input_data
            l0_h2d = pir.h2d_stream((COMP_BATCH, ),
                                    pir.uint32,
                                    name="l0_stream")
            l0 = ops.host_load(l0_h2d, "d")
            data[l0_h2d.tensor_id()] = label_data

            var_shard_shape: Tuple[int, ...] = (weight_data.size //
                                                NUM_LOCAL_REPLICAS, )

            w = pir.variable(weight_data, name="w_full")
            remote_buffer = RemoteBuffer(ir,
                                         var_shard_shape,
                                         dtypes.float32,
                                         entries=NUM_LOCAL_REPLICAS)

            remote_arg = pir.remote_replica_sharded_variable(
                w, remote_buffer, 0)
            loaded_w = ops.remote_load(remote_buffer, remote_arg)

            full_w = ops.collectives.replicated_all_gather(loaded_w).reshape_(
                (4, 4))

            fwd = self.Linear()
            fwd_graph = ir.create_graph(fwd, d0)
            fwd_call_info = ops.call_with_info(
                fwd_graph, d0, subgraph_in_to_parent_in={fwd.W: full_w})

            y = fwd_call_info.get_output_tensors()[0]

            y_d2h = pir.d2h_stream(y.shape, pir.float32, name="y_stream")
            ops.host_store(y_d2h, y)

            probs = ops.softmax(y, 0)
            loss, dx = ops.nll_loss_with_softmax_grad(probs, l0, loss_grad=1)

        ### Backward pass ###
        bwd_info = pir.transforms.autodiff(fwd_graph)
        bwd_graph = bwd_info.graph

        with main:
            tensors_required_for_bwd = bwd_info.get_inputs_from_forward_call_info(
                fwd_call_info)
            bwd_call_info = ops.call_with_info(
                bwd_graph,
                dx,
                subgraph_in_to_parent_in=tensors_required_for_bwd)

        expected_outputs = bwd_info.expected_outputs
        d0_grad: Tensor = None
        w0_grad: Tensor = None

        sg_d0 = fwd_call_info.op_in_to_subgraph_in_tensor(d0)
        sg_w0 = fwd_call_info.op_in_to_subgraph_in_tensor(full_w)

        def get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                ec: ExpectedConnection) -> Tensor:
            # If (t, FwdGrad) appears at index i in expected_outputs, it is
            # guaranteed that t' (the grad of t) appears at output index i in the
            # grad graph.
            sg_out_idx = expected_outputs.index(ec)
            op_out_idx = bwd_call_info.subgraph_in_to_op_in_index(sg_out_idx)
            parent_grad = bwd_call_info.get_op_output_tensor(op_out_idx)

            return parent_grad

        for ec in expected_outputs:
            # Should always be the case for expected_outputs
            assert ec.connection_type == ExpectedConnectionType.FwdGrad

            sg_fwd_tensor = ec.fwd_tensor

            if sg_fwd_tensor == sg_d0:
                d0_grad = get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                    ec)
            elif sg_fwd_tensor == sg_w0:
                w0_grad = get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                    ec)

        ### Weight update ###
        # Note the pir.in_sequence() : forces ops in the correct order.
        with main, pir.in_sequence():
            grad_shard: Tensor = ops.collectives.replicated_reduce_scatter(
                w0_grad,
                op='add',
                configure_output_for_replicated_tensor_sharding=True)
            ops.var_updates.accumulate_(loaded_w, grad_shard)
            ops.remote_store(remote_buffer, remote_arg, loaded_w)

            w_d2h = pir.d2h_stream(loaded_w.shape, loaded_w.dtype)
            ops.host_store(w_d2h, loaded_w)

            grad_shard_d2h = pir.d2h_stream(grad_shard.shape, grad_shard.dtype)
            ops.host_store(grad_shard_d2h, grad_shard)

        ir._pb_ir.logIr()

        anchors = self.run(
            ir, data, main,
            [y_d2h.tensor_id(),
             w_d2h.tensor_id(),
             grad_shard_d2h.tensor_id()])

        np_loaded_w = anchors[w_d2h.tensor_id()]
        np_grad_shard = anchors[grad_shard_d2h.tensor_id()]

        # Check the weight has updated. So w = weight_data + w'
        assert np.allclose(np_loaded_w, weight_data + np_grad_shard)

    def run(self, ir: pir.Ir, data: Mapping[str, np.ndarray], main: pir.Graph,
            anchor_ids: List[str]):
        dataFlow = popart.DataFlow(
            BPS, {id: popart.AnchorReturnType("All")
                  for id in anchor_ids})
        ir._pb_ir.setDataFlow(dataFlow)

        opts = ir._pb_ir.getSessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = 4
        if ACCL_FACTOR > 1:
            opts.enableGradientAccumulation = True
            opts.accumulationFactor = ACCL_FACTOR

        ir._pb_ir.updateVertices()
        p = _ir.patterns.Patterns(_ir.patterns.PatternsLevel.Minimal)
        ir._pb_ir.setPatterns(p)
        for _g in ir._pb_ir.getAllGraphs():
            ir._pb_ir.applyPreAliasPatterns(_g)

        device = tu.create_test_device(opts.replicatedGraphCount)

        session = popart.InferenceSession.fromIr(ir=ir._pb_ir,
                                                 deviceInfo=device)

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO(data, anchors)

        session.run(stepio)
        session.weightsToHost()

        return anchors
