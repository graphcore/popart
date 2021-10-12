# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popart._internal.ir as _ir
import popart.ir as pir
from popart.ir.tensor import Variable, Constant

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
            exp_dtype = dtype if dtype is not None else pir.float32

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

    @pytest.mark.parametrize('t_class', [Variable, Constant])
    def test_construction1(self, t_class):
        """Test construction of tensors that hold 0-d data at graph creation."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            t = ctor_map[t_class](1)
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
