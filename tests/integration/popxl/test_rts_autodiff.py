# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

import popxl
import popxl.dtypes as dtypes
import popxl.ops as ops
from popxl.remote_buffer import RemoteBuffer
from popxl.tensor import Tensor
from popxl.transforms.autodiff import ExpectedConnection, ExpectedConnectionType
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
BATCH_SIZE = NUM_LOCAL_REPLICAS * ACCL_FACTOR

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
    class Linear(popxl.Module):
        def __init__(self):
            self.W: Tensor = None

        def build(self, d0: Tensor) -> Tensor:
            self.W = popxl.graph_input((4, 4), popxl.float32, "w0")
            y = d0 @ self.W
            return y

    @tu.requires_ipu_model
    def test_rts(self):

        ir = popxl.Ir()
        data: Mapping[popxl.HostToDeviceStream, np.ndarray] = {}
        main = ir.main_graph
        ir.replication_factor = NUM_LOCAL_REPLICAS
        ### Forward pass ###
        with main:
            d0_h2d = popxl.h2d_stream((COMP_BATCH, HIDDEN_SIZE),
                                      popxl.float32,
                                      name="d0_stream")
            d0 = ops.host_load(d0_h2d, "d")

            data[d0_h2d] = input_data.reshape(
                (BATCH_SIZE, COMP_BATCH, HIDDEN_SIZE))
            l0_h2d = popxl.h2d_stream((COMP_BATCH, ),
                                      popxl.uint32,
                                      name="l0_stream")
            l0 = ops.host_load(l0_h2d, "d")
            data[l0_h2d] = label_data.reshape((
                BATCH_SIZE,
                COMP_BATCH,
            ))

            var_shard_shape: Tuple[int, ...] = (weight_data.size //
                                                NUM_LOCAL_REPLICAS, )
            remote_buffer = RemoteBuffer(var_shard_shape,
                                         dtypes.float32,
                                         entries=NUM_LOCAL_REPLICAS)
            w = popxl.remote_replica_sharded_variable(weight_data,
                                                      remote_buffer, 0)
            loaded_w = ops.remote_load(remote_buffer, 0)
            full_w = ops.collectives.replicated_all_gather(loaded_w).reshape_(
                (4, 4))

            fwd = self.Linear()
            fwd_graph = ir.create_graph(fwd, d0)
            fwd_call_info = ops.call_with_info(fwd_graph,
                                               d0,
                                               inputs_dict={fwd.W: full_w})

            y = fwd_call_info.outputs[0]

            y_d2h = popxl.d2h_stream(y.shape, popxl.float32, name="y_stream")
            ops.host_store(y_d2h, y)

            probs = ops.softmax(y, 0)
            loss, dx = ops.nll_loss_with_softmax_grad(probs, l0, loss_grad=1)

        ### Backward pass ###
        bwd_info = popxl.transforms.autodiff(fwd_graph)
        bwd_graph = bwd_info.graph

        with main:
            tensors_required_for_bwd = bwd_info.inputs_dict(fwd_call_info)
            bwd_call_info = ops.call_with_info(
                bwd_graph, dx, inputs_dict=tensors_required_for_bwd)

        expected_outputs = bwd_info.expected_outputs
        d0_grad: Tensor = None
        w0_grad: Tensor = None

        sg_d0 = fwd_call_info.parent_to_graph(d0)
        sg_w0 = fwd_call_info.parent_to_graph(full_w)

        def get_grad_tensor_in_main_graph_from_fwdgrad_expected_connection(
                ec: ExpectedConnection) -> Tensor:
            # If (t, FwdGrad) appears at index i in expected_outputs, it is
            # guaranteed that t' (the grad of t) appears at output index i in the
            # grad graph.
            sg_out_idx = expected_outputs.index(ec)
            op_out_idx = bwd_call_info.graph_to_parent_input_index(sg_out_idx)
            parent_grad = bwd_call_info.parent_output(op_out_idx)

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
        # Note the popxl.in_sequence() : forces ops in the correct order.
        with main, popxl.in_sequence():
            grad_shard: Tensor = ops.collectives.replicated_reduce_scatter(
                w0_grad,
                op='add',
                configure_output_for_replicated_tensor_sharding=True)
            ops.var_updates.accumulate_(loaded_w, grad_shard)
            ops.remote_store(remote_buffer, 0, loaded_w)

            # Need to gather the weight and gradient, since sharded tensors
            # cannot be stored to the host via streams
            loaded_gathered_w = ops.collectives.replicated_all_gather(loaded_w)

            grad = ops.collectives.replicated_all_gather(grad_shard)

            w_d2h = popxl.d2h_stream(loaded_gathered_w.shape,
                                     loaded_gathered_w.dtype)
            ops.host_store(w_d2h, loaded_gathered_w)

            grad_d2h = popxl.d2h_stream(grad.shape, grad.dtype)
            ops.host_store(grad_d2h, grad)

        ir._pb_ir.logIr()

        session, outputs = self.run(ir, data)

        np_loaded_gathered_w = outputs[w_d2h]
        np_grad = outputs[grad_d2h]

        # Check the weight has updated. So w = weight_data + w'
        np.testing.assert_allclose(np_loaded_gathered_w, weight_data + np_grad)
        # w now has been updated as we have synced remote buffers with device.
        np.testing.assert_allclose(session.get_tensor_data(w),
                                   np_loaded_gathered_w[0])

    def run(self, ir: popxl.Ir,
            data: Mapping[popxl.HostToDeviceStream, np.ndarray]):

        ir.num_host_transfers = 1
        ir.replication_factor = NUM_LOCAL_REPLICAS

        session = popxl.Session(ir, "ipu_model")

        outputs = session.run(data)

        return session, outputs
