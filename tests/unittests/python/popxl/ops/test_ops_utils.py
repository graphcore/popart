# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
from popxl.ops.utils import check_tensor_ipu_and_tile_set
import popxl


class TestCheckTensorIpuAndTileSet:
    def test_same_ipus_and_tile_set(self):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t1 = popxl.variable(0) + 0
            t2 = popxl.variable(0) + 0
            t3 = popxl.variable(0) + 0

        check_tensor_ipu_and_tile_set(t1=t1, t2=t2, t3=t3)

    def test_different_ipus(self):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t1 = popxl.variable(0) + 0
            with popxl.ipu(1):
                t2 = popxl.variable(0) + 0
            t3 = popxl.variable(0) + 0

        with pytest.raises(ValueError) as excinfo:
            check_tensor_ipu_and_tile_set(t1=t1, t2=t2, t3=t3)

        assert 't2' in str(excinfo.value)
        assert 'IPUs' in str(excinfo.value)

    def test_different_tile_sets(self):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t1 = popxl.variable(0) + 0
            t2 = popxl.variable(0) + 0
            with popxl.io_tiles():
                t3 = popxl.variable(0) + 0

        with pytest.raises(ValueError) as excinfo:
            check_tensor_ipu_and_tile_set(t1=t1, t2=t2, t3=t3)

        assert 't3' in str(excinfo.value)
        assert 'tile sets' in str(excinfo.value)

    def test_one_missing_but_ok(self):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t1 = popxl.variable(0) + 0
            t2 = popxl.variable(0) + 0
            t3 = popxl.variable(0)  # Undefined IPU

        check_tensor_ipu_and_tile_set(t1=t1, t2=t2, t3=t3)

    def test_one_missing_but_different_tile_sets(self):
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t1 = popxl.variable(0) + 0
            with popxl.io_tiles():
                t2 = popxl.variable(0) + 0
            t3 = popxl.variable(0)  # Undefined IPU/tileset

        with pytest.raises(ValueError) as excinfo:
            check_tensor_ipu_and_tile_set(t1=t1, t2=t2, t3=t3)

        assert 't2' in str(excinfo.value)
        assert 'tile sets' in str(excinfo.value)
