# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.core.numeric import indices
import pytest

import popart._internal.ir as _ir
import popart.ir as pir
from popart.ir.tensor import Variable, Constant, downcast_np_dtypes
from popart.ir.errors import UndefinedValue

type_map = {Variable: _ir.TensorType.Variable, Constant: _ir.TensorType.Const}
ctor_map = {Variable: pir.variable, Constant: pir.constant}


class TestTensor:
    @pytest.mark.parametrize('t_class', [Variable, Constant])
    @pytest.mark.parametrize('data', [
        np.random.rand(1, 2, 3),
        [[[1, 2, 3], [4, 5, 6]]],
        (((1, 2, 3), (4, 5, 6)), ),
    ])
    @pytest.mark.parametrize('dtype', [pir.float16, None])
    @pytest.mark.parametrize('name', ['a', None])
    def test_construction0(self, t_class, data, dtype, name):
        """Test construction of tensors that hold n-d data at graph creation."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            kwargs = {}
            if name is not None:
                kwargs['name'] = name
            if dtype is not None:
                kwargs['dtype'] = dtype

            exp_name = f'{name}' if name is not None else 't'

            def exp_np_dtype_(dtype):
                if dtype is not None:
                    return dtype
                else:
                    np_data = np.array(data)
                    if np_data.dtype in downcast_np_dtypes:
                        return pir.dtypes.dtype.as_dtype(
                            downcast_np_dtypes[np_data.dtype])
                    else:
                        return pir.dtypes.dtype.as_dtype(np_data.dtype)

            exp_dtype = exp_np_dtype_(dtype)

            t = ctor_map[t_class](data, **kwargs)

            assert isinstance(t, t_class)
            assert t.dtype == exp_dtype
            assert t.shape == (1, 2, 3)
            assert len(t) == 1
            assert t.nelms == 6
            assert t.name == exp_name

            pb_t: _ir.Tensor = main._pb_graph.getTensor(exp_name)
            assert pb_t.id == t.name
            assert pb_t.id == t.id
            assert pb_t.tensorType() == type_map[t_class]
            assert pb_t.hasTensorData()
            hash(t)

    @pytest.mark.parametrize('t_class', [Variable, Constant])
    def test_construction1(self, t_class):
        """Test construction of tensors that hold 0-d data at graph creation."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            t = ctor_map[t_class](1.0)
            assert t.dtype == pir.float32
            assert t.shape == ()
            assert t.nelms == 1

    def test__ensure_tensor(self):
        """Test the `_ensure_tensor()` method."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)
            b = pir.variable(2)
            c = a._ensure_tensor(b)
            d = a._ensure_tensor(3)

            assert c == b
            assert isinstance(d, Constant)
            assert d.dtype == a.dtype

    def test_from_pb_type(self):
        """Test the from_pb_tensor returns the correct python type"""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)
            c = pir.constant(2)

        assert isinstance(a, Variable)
        new_a = pir.Tensor._from_pb_tensor(a._pb_tensor)
        assert isinstance(new_a, Variable)
        assert isinstance(c, Constant)
        new_c = pir.Tensor._from_pb_tensor(c._pb_tensor)
        assert isinstance(new_c, Constant)

    def test_get_ir(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)
            assert a.ir() == ir

    def test_cmp(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)
            b = pir.variable(1)
            assert a != b  # test __eq__
            assert len(set([a, b])) == 2  # test __hash__
            str(a)  # test __repr__

    def test_iter_dunder(self):
        with pir.Ir().main_graph():
            x = pir.variable(0)
            l = []
            with pytest.raises(TypeError):
                l += x

    def test_contains_dunder(self):
        with pir.Ir().main_graph():
            x = pir.variable(0)
            with pytest.raises(TypeError):
                1 in x


class TestTensorIpuAndTileSet:
    def test_ipu_undefined(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)

            with pytest.raises(UndefinedValue):
                a.ipu

    def test_ipu_defined_default(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1) + 0
            assert a.ipu == 0

    def test_ipu_set(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            with pir.ipu(1):
                a = pir.variable(1) + 0
            assert a.ipu == 1

    def test_tile_set_undefined(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1)

            with pytest.raises(UndefinedValue):
                a.tile_set

    def test_tile_set_compute(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.variable(1) + 0
            assert a.tile_set == 'compute'

    def test_ipu_defined_default(self):
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            with pir.io_tiles():
                a = pir.variable(1) + 0
            assert a.tile_set == 'io'


class TestTensorGetItem:
    def test_integer_slicing(self):
        with pir.Ir().main_graph():
            x = pir.variable(np.arange(10))
            y = x[1]
            assert y.shape == tuple()  # Empty as dim squeezed

    def test_slice_slicing(self):
        with pir.Ir().main_graph():
            x = pir.variable(np.random.rand(10, 10))
            y = x[1:3]
            assert y.shape == (2, 10)

    def test_both_int_and_slice_slicing(self):
        with pir.Ir().main_graph():
            x = pir.variable(np.random.rand(10, 10))
            y = x[1:3, 2]
            assert y.shape == (2, )

    @pytest.mark.parametrize('tensorlike', [pir.variable, np.array, list])
    def test_integer_indexing_tensor(self, tensorlike):
        with pir.Ir().main_graph():
            indices = [[0, 1], [1, 1]]
            indices = tensorlike(indices)
            x = pir.variable(np.random.rand(10, 10))
            y = x[indices]
            assert y.shape == (2, 2, 10)

    @pytest.mark.parametrize('tensorlike', [pir.variable, np.array, list])
    def test_bool_indexing_tensor(self, tensorlike):
        with pir.Ir().main_graph():
            mask = [[True, False], [True, False], [False, True], [True, True]]
            mask = tensorlike(mask)
            x = pir.variable(np.random.rand(4, 2))
            y = x[mask]
            assert y.shape == (4, 2)

    @pytest.mark.parametrize('tensorlike', [pir.variable, np.array, list])
    def test_bool_indexing_tensor_broadcast(self, tensorlike):
        with pir.Ir().main_graph():
            mask = [[True], [True], [False], [True]]
            mask = tensorlike(mask)
            x = pir.variable(np.random.rand(4, 2))
            y = x[mask]
            assert y.shape == (4, 2)

    @pytest.mark.parametrize("key", ['a', True, 1.1])
    def test_bad_key(self, key):
        with pir.Ir().main_graph():
            x = pir.variable(np.arange(2))
            with pytest.raises(TypeError):
                y = x[key]
