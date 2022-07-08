# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popxl


def test_context_manager():
    ir = popxl.Ir()
    main = ir.main_graph

    with pytest.raises(RuntimeError):
        popxl.gcg()

    with pytest.raises(RuntimeError):
        popxl.gmg()

    with main:
        assert popxl.gcg() == main
        assert popxl.gmg() == main
        sg = ir.create_empty_graph("sg")
        with sg:
            assert popxl.gcg() == sg
            assert popxl.gmg() == main
            assert sg.main_graph == main

    with pytest.raises(RuntimeError):
        popxl.gcg()

    with pytest.raises(RuntimeError):
        popxl.gmg()


def test_nested_irs():
    mg1 = popxl.Ir().main_graph
    mg2 = popxl.Ir().main_graph

    # You cant test graphs with different irs
    with pytest.raises(RuntimeError):
        with mg1:
            with mg2:
                pass


def test_get_main_graph():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        sg = ir.create_empty_graph("sg")

    # Main graph does not necessarily need to be top of context
    with sg:
        assert popxl.gmg() == main
