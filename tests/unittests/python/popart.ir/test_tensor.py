# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popart._internal.ir as _ir
import popart.ir as pir
from popart import popart_exception

type_map = {
    pir.Variable: _ir.TensorType.Variable,
    pir.Constant: _ir.TensorType.Const
}
ctor_map = {pir.Variable: pir.variable, pir.Constant: pir.constant}


class TestTensor:
    @pytest.mark.parametrize('t_class', [pir.Variable, pir.Constant])
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
            assert t.nelms == 6
            assert t.name == exp_name

            pb_t: _ir.Tensor = main._pb_graph.getTensor(exp_name)
            assert pb_t.id == t.name
            assert pb_t.id == t.id
            assert pb_t.tensorType() == type_map[t_class]
            assert pb_t.hasTensorData()

    @pytest.mark.parametrize('t_class', [pir.Variable, pir.Constant])
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
            assert isinstance(d, pir.Constant)
            assert d.dtype == a.dtype