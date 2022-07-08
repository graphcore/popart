# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Mapping, Tuple

import numpy as np
import pytest
import popxl
import popxl.ops as ops
from popart import InferenceSession


class TestIOSizes:
    def test_uneven_host_loads(self):
        """Check we can run :
        h2d_stream_1 = ...
        h2d_stream_2 = ...

        # repeat op in pseudocode
        for i in range(batches_per_step):
            a = ops.host_load(h2d_stream_1)

            # Intentionally load twice
            b = ops.host_load(h2d_stream_2)
            b_2 = ops.host_load(h2d_stream_2) # <-same stream

        and set BPS to twice repeat iters, whilst also providing the double amount of data
        for BOTH streams.

        TODO: T56405 This is a hack and should be fixed.

        """
        BPS = 10
        NUM_LOCAL_REPLICAS = 1
        DATA_SHAPE = 4

        input_data = []

        for i in range(0, BPS * NUM_LOCAL_REPLICAS):
            input_data += [np.random.rand(DATA_SHAPE, DATA_SHAPE).astype(np.float32)]

        input_data: np.ndarray = np.concatenate(input_data)

        ir = popxl.Ir()
        data: Mapping[popxl.HostToDeviceStream, np.ndarray] = {}
        main = ir.main_graph

        with main:
            repeat_sg = ir.create_empty_graph("repeat")
            d0_h2d = popxl.h2d_stream(
                (DATA_SHAPE, DATA_SHAPE), popxl.float32, name="d0_stream"
            )
            e0_h2d = popxl.h2d_stream(
                (DATA_SHAPE, DATA_SHAPE), popxl.float32, name="e0_stream"
            )
            out_d2h = popxl.d2h_stream(
                (DATA_SHAPE, DATA_SHAPE), popxl.float32, name="out_d2h_stream"
            )
            with repeat_sg, popxl.in_sequence():

                d0 = ops.host_load(d0_h2d, "d")
                d0 = ops.print_tensor(d0)

                input_copy = (
                    input_data.copy()
                    .reshape((BPS, DATA_SHAPE, DATA_SHAPE))
                    .astype(np.float32)
                )

                data[d0_h2d] = np.concatenate([input_copy] * 2, axis=0)
                data[e0_h2d] = np.concatenate([input_copy] * 2, axis=0)

                # This fails, batch 5 (zero index)
                # Unable to fetch input data from IStepIO
                e0 = ops.host_load(e0_h2d, "e0")
                e0 = ops.print_tensor(e0)

                ### Load from this buffer twice! ###
                e1 = ops.host_load(e0_h2d, "e1")
                e1 = ops.print_tensor(e1)

                out = d0 + e0 + e1

                ops.host_store(out_d2h, out)

            ops.repeat(repeat_sg, repeat_count=BPS)

        ir.num_host_transfers = BPS * 2
        ir.replication_factor = NUM_LOCAL_REPLICAS

        session = popxl.Session(ir, "ipu_model")
        for k, v in data.items():
            print(k, k.tensor_id, "data shape:", v.shape)

        outputs = {}
        outputs[out_d2h] = np.zeros((BPS * 2, DATA_SHAPE, DATA_SHAPE)).astype(
            np.float32
        )

        with session:
            session.run_with_outputs(data, outputs)

        for k, v in outputs.items():
            print(k, ":", v.shape)

            for i in range(BPS):
                print("Batch:", i)
                d0_data = data[d0_h2d][i, ...]
                # e0 data advances 2 batches per iteration
                e0_data = data[e0_h2d][2 * i, ...]
                # e1 data should have advanced by 1 batch vs e0:
                e1_data = data[e0_h2d][2 * i + 1, ...]

                assert np.allclose(v[i, ...], d0_data + e0_data + e1_data)

    @pytest.mark.parametrize("data_size", [[1], [4, 4], [4]])
    @pytest.mark.parametrize("replication_factor", [1, 2, 4])
    @pytest.mark.parametrize("num_host_transfers", [2, 32])
    def test_io_sizes(
        self,
        monkeypatch,
        data_size: Tuple[int],
        replication_factor: int,
        num_host_transfers: int,
    ):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            repeat_sg = ir.create_empty_graph("repeat")
            d0_h2d = popxl.h2d_stream(data_size, popxl.float32, name="d0_stream")
            out_d2h = popxl.d2h_stream(data_size, popxl.float32, name="out_d2h_stream")
            with repeat_sg, popxl.in_sequence():

                d0 = ops.host_load(d0_h2d, "d")
                d0 = ops.gelu(d0)

                ops.host_store(out_d2h, d0)

            ops.repeat(repeat_sg, repeat_count=num_host_transfers)

        ir.num_host_transfers = num_host_transfers
        ir.replication_factor = replication_factor

        def dummy_function(self, *args, **kwargs):  # pylint: disable=unused-argument
            # Do nothing
            pass

        # Monkeypatch prepareDevice / weightsFromHost to do nothing, this way we avoid the overhead
        # of preparing the device every time for this test. Note: weightsFromHost will segfault
        # if called so we have to monkey patch it too.
        monkeypatch.setattr(InferenceSession, "prepareDevice", dummy_function)

        monkeypatch.setattr(InferenceSession, "weightsFromHost", dummy_function)

        session = popxl.Session(ir, "ipu_model")

        input_shape = (num_host_transfers, replication_factor, *data_size)

        data = {}
        data[d0_h2d] = np.zeros(shape=input_shape).astype(np.float32)
        # Assume the user squeezes their inputs, unless they have size [1] data:
        if data_size != [1]:
            data[d0_h2d] = data[d0_h2d].squeeze()

        outputs = {}
        outputs[out_d2h] = np.zeros(shape=input_shape).astype(np.float32)
        if data_size != [1]:
            outputs[out_d2h] = outputs[out_d2h].squeeze()

        session._validate_run_inputs(data)
        session._validate_run_outputs(outputs)
