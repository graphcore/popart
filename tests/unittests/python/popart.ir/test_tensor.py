# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popart._internal.ir as _ir
import popart.ir as pir
from popart import popart_exception


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

        type_map = {
            pir.Variable: _ir.TensorType.Variable,
            pir.Constant: _ir.TensorType.Const
        }

        with main:
            kwargs = {}
            if name is not None:
                kwargs['name'] = name
            if dtype is not None:
                kwargs['dtype'] = dtype

            exp_name = f'{name}_0' if name is not None else 't_0'
            exp_dtype = dtype if dtype is not None else pir.float32

            t = t_class(data, **kwargs)

            assert t.dtype == exp_dtype
            assert t.shape == (1, 2, 3)
            assert t.dim == 3
            assert t.nelms == 6
            assert t.name == exp_name
            assert t._graph == main

            pb_t: _ir.Tensor = t._graph._pb_graph.getTensor(exp_name)
            assert pb_t.id == t.name
            assert pb_t.id == t._pb_tensor_id
            assert pb_t.tensorType() == type_map[t_class]
            assert pb_t.hasTensorData() == True

    @pytest.mark.parametrize('t_class', [pir.Variable, pir.Constant])
    def test_construction1(self, t_class):
        """Test construction of tensors that hold 0-d data at graph creation."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            t = t_class(1)
            assert t.dtype == pir.float32
            assert t.shape == ()
            assert t.dim == 0
            assert t.nelms == 1

    @pytest.mark.parametrize('dtype', [pir.float16, pir.float32])
    @pytest.mark.parametrize('name', ['a', None])
    def test_construction2(self, name, dtype):
        """Test construction of placeholder tensors."""
        ir = pir.Ir()
        main = ir.main_graph()

        exp_name = f'{name}_0' if name is not None else 't_0'
        exp_dtype = dtype if dtype is not None else pir.float32

        with main:
            kwargs = {'name': name} if name is not None else {}
            t = pir.Placeholder(dtype, (1, 2, 3), **kwargs)

            assert t.dtype == exp_dtype
            assert t.shape == (1, 2, 3)
            assert t.dim == 3
            assert t.nelms == 6
            assert t.name == exp_name
            assert t._graph == main

            # No tensor is created in the IR yet.
            with pytest.raises(popart_exception) as excinfo:
                t._graph._pb_graph.getTensor(exp_name)
            exp_prefix = 'No Ir::Tensor with TensorId'
            assert excinfo.value.args[0].startswith(exp_prefix)

    def test__ensure_tensor(self):
        """Test the `_ensure_tensor()` method."""
        ir = pir.Ir()
        main = ir.main_graph()

        with main:
            a = pir.Variable(1)
            b = pir.Variable(2)
            c = a._ensure_tensor(b)
            d = a._ensure_tensor(3)

            assert c == b
            assert isinstance(d, pir.Constant)
            assert d.dtype == a.dtype
