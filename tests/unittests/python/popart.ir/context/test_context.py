# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart.ir as pir


def test_context_manager():
    ir = pir.Ir()
    main = ir.main_graph()

    with pytest.raises(RuntimeError) as excinfo:
        pir.gcg()

    with pytest.raises(RuntimeError) as excinfo:
        pir.gmg()

    with main:
        assert pir.gcg() == main
        assert pir.gmg() == main
        sg = ir.create_empty_graph('sg')
        with sg:
            assert pir.gcg() == sg
            assert pir.gmg() == main
            assert sg.get_main_graph() == main

    with pytest.raises(RuntimeError) as excinfo:
        pir.gcg()

    with pytest.raises(RuntimeError) as excinfo:
        pir.gmg()


def test_nested_irs():
    mg1 = pir.Ir().main_graph()
    mg2 = pir.Ir().main_graph()

    # You cant test graphs with different irs
    with pytest.raises(RuntimeError):
        with mg1:
            with mg2:
                pass


def test_get_main_graph():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        sg = ir.create_empty_graph('sg')

    # Main graph does not necessarily need to be top of context
    with sg:
        assert pir.gmg() == main
