# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np


class TestGreater:
    def test_greater(self):
        '''Compute element-wise where the first tensor is greater than the second tensor.'''
        i1 = np.array([1.1, 3.7, 8.4, 2.2], dtype=np.float32)
        i2 = np.array([0.1, 6.7, 3.4, 2.2], dtype=np.float32)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([4], popxl.float32, name="in_stream_0")
            input1 = popxl.h2d_stream([4], popxl.float32, name="in_stream_1")
            a = ops.host_load(input0, "a")
            b = ops.host_load(input1, "b")
            o = ops.greater(a, b)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: i1, input1: i2})
        # greater in numpy
        np_outputs = np.greater(i1, i2)
        # compare the result between PopXL and numpy
        assert (outputs[o_d2h] == np_outputs).all()

    def test_greater_broadcast_1(self):
        '''Broadcast on the row and computes where the first tensor is greater than the second tensor.'''
        i1 = np.array([[1.1, 3.7, 8.4, 2.2], [2.8, 3.4, 8.0, 4.2]], dtype=np.float32)
        i2 = np.array([[0.1, 6.7, 3.4, 2.2]], dtype=np.float32)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([2, 4], popxl.float32, name="in_stream_0")
            input1 = popxl.h2d_stream([1, 4], popxl.float32, name="in_stream_1")
            a = ops.host_load(input0, "a")
            b = ops.host_load(input1, "b")
            o = ops.greater(a, b)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: i1, input1: i2})
        # greater in numpy
        np_outputs = np.greater(i1, i2)
        # compare the result between PopXL and numpy
        assert (outputs[o_d2h] == np_outputs).all()

    def test_greater_broadcast_2(self):
        '''Broadcast on the column and computes where the first tensor is greater than the second tensor.'''
        i1 = np.array([[1.1, 3.7, 8.4, 2.2], [2.8, 3.4, 8.0, 4.2]], dtype=np.float32)
        i2 = np.array([[3.7], [3.4]], dtype=np.float32)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([2, 4], popxl.float32, name="in_stream_0")
            input1 = popxl.h2d_stream([2, 1], popxl.float32, name="in_stream_1")
            a = ops.host_load(input0, "a")
            b = ops.host_load(input1, "b")
            o = ops.greater(a, b)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: i1, input1: i2})
        # greater in numpy
        np_outputs = np.greater(i1, i2)
        # compare the result between PopXL and numpy
        assert (outputs[o_d2h] == np_outputs).all()

    def test_greater_broadcast_3(self):
        '''
        Broadcast on the row and column and computes where the first tensor is greater
        than the second tensor with fp16 data type.
        '''
        i1 = np.array([[1.1, 3.7, 8.4, 2.2], [2.8, 3.4, 8.0, 4.2]], dtype=np.float16)
        i2 = np.array([3.7], dtype=np.float16)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([2, 4], popxl.float16, name="in_stream_0")
            input1 = popxl.h2d_stream([1], popxl.float16, name="in_stream_1")
            a = ops.host_load(input0, "a")
            b = ops.host_load(input1, "b")
            o = ops.greater(a, b)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: i1, input1: i2})
        # greater in numpy
        np_outputs = np.greater(i1, i2)
        # compare the result between PopXL and numpy
        assert (outputs[o_d2h] == np_outputs).all()
