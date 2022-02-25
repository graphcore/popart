# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
from typing import List

import numpy as np
import popart
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import pytest
from popart.ir import Tensor

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / "integration"))
import test_util as tu

_BATCH_SIZE = 6
_IN_FEATURES = 8

_IN_SHAPE = (_BATCH_SIZE, _IN_FEATURES)


def host_store_and_return_d2h_stream(g: pir.Graph,
                                     t_: Tensor) -> pir.DeviceToHostStream:
    """Create a host store op with the provided graph and return the stream.

    Args:
        g (pir.Graph): The graph to use
        t_ (Tensor): The tensor to host load.

    Returns:
        pir.DeviceToHostStream: The host stream created.
    """
    with g:
        d2h = pir.d2h_stream(t_.shape, t_.dtype, name=t_.name + "_stream")
        ops.host_store(d2h, t_)
    return d2h


def test_inplacing_ambiguity_0():
    """Simple example of an inplacing ambiguity.
    w0 : variable
    w1 : variable
    w0 <- relu_(w0) (1)
    w0 <- w0 + w1 (2)

    With no constraints between (1) and (2) the final value of w0 is ambiguous.

    We check this is detected in this test.
    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = pir.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = pir.variable(w1_data, name="w1")

        m: Tensor = ops.relu_(w0)
        w0 += w1

    m_d2h = host_store_and_return_d2h_stream(main, m)

    session = setup_ir(ir, [m_d2h])

    with pytest.raises(popart.popart_exception) as e_info:
        session.checkInplacingAmbiguity()
    assert (e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph():
    """ Check there is the same ambiguity in a called subgraph"""

    class InplaceGraph(pir.Module):
        def __init__(self):
            pass

        def build(self, w0: Tensor, w1: Tensor):
            m: Tensor = ops.relu_(w0)
            w0 += w1
            return m, w0

    ir = pir.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = pir.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = pir.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, w0, w1)
        w0, m = ops.call(inplace_graph, w0, w1)

        m_d2h = host_store_and_return_d2h_stream(main, m)

        session = setup_ir(ir, [m_d2h])

        with pytest.raises(popart.popart_exception) as e_info:
            session.checkInplacingAmbiguity()
        assert (
            e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph_1():
    """Same as above, but for a weight passed as a subgraph input"""

    class InplaceGraph(pir.Module):
        def __init__(self):
            self.w1: pir.Tensor = None

        def build(self, w0: Tensor):
            self.w1 = pir.graph_input(w0.shape, w0.dtype, "w1")
            m: Tensor = ops.relu_(w0)
            w0 += self.w1
            return m, w0

    ir = pir.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = pir.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = pir.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, w0)
        w0, m = ops.call(inplace_graph, w0, inputs_dict={inplace_.w1: w1})

        m_d2h = host_store_and_return_d2h_stream(main, m)

        session = setup_ir(ir, [m_d2h])

        with pytest.raises(popart.popart_exception) as e_info:
            session.checkInplacingAmbiguity()
        assert (
            e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph_2():
    """As above but with a host load, rather than variable."""

    class InplaceGraph(pir.Module):
        def __init__(self):
            self.w1: pir.Tensor = None

        def build(self, x0: Tensor):
            self.w1 = pir.graph_input(x0.shape, x0.dtype, "w1")
            m: Tensor = ops.relu_(x0)
            x0 += self.w1
            return m, x0

    ir = pir.Ir()

    main = ir.main_graph
    with main:
        input_ = pir.h2d_stream(_IN_SHAPE, pir.float32, name="input_stream")
        x0 = ops.host_load(input_, "x0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = pir.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, x0)
        x0, m = ops.call(inplace_graph, x0, inputs_dict={inplace_.w1: w1})

        m_d2h = host_store_and_return_d2h_stream(main, m)

        session = setup_ir(ir, [m_d2h])

        with pytest.raises(popart.popart_exception) as e_info:
            session.checkInplacingAmbiguity()
        assert (
            e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_false_positive():
    """Simple example of a false positive inplacing ambiguity.
   * a <- variable;
   * b <- variable;
   * d <- a.add_(5); // a's value changes.
   * d <- a.copyFrom_(b); // copy from b to a.
   * e <- d.add(5);

   The issue with this case is that poprithms does not distinguish
   between updates based on existing values, and updates to completely new
   values.
    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        b = pir.variable(b_data, name="b")

        d = a + 5
        d = ops.var_updates.copy_var_update_(a, b)
        e = d + 5

    d_d2h = host_store_and_return_d2h_stream(main, e)
    e_d2h = host_store_and_return_d2h_stream(main, e)

    session = setup_ir(ir, [d_d2h, e_d2h])

    with pytest.raises(popart.popart_exception) as e_info:
        session.checkInplacingAmbiguity()
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_false_positive_2():
    """Simple example of a false positive inplacing ambiguity. Poprithms can't tell that
    relu_ is idempotent.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.relu_(); // if this were gelu_, there would be a genuine ambiguity,
    as `relu(a) == relu(relu(a))` but `gelu(relu(a)) != relu(gelu(a)) != relu(a)`.

    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.relu_(
            a)  #if this were gelu_, there would be a genuine ambiguity.

    b_d2h = host_store_and_return_d2h_stream(main, b)
    c_d2h = host_store_and_return_d2h_stream(main, c)

    session = setup_ir(ir, [b_d2h, c_d2h])

    with pytest.raises(popart.popart_exception) as e_info:
        session.checkInplacingAmbiguity()
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_3():
    """As above but the second relu_ is now a gelu_ which causes a genuine ambiguity.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.gelu_(); // This is now a genuine ambiguity.
     See `test_inplacing_ambiguity_false_positive_2` for why.

    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        b_d2h = pir.d2h_stream(_IN_SHAPE, pir.dtypes.float32, name="b_stream")
        c_d2h = pir.d2h_stream(_IN_SHAPE, pir.dtypes.float32, name="c_stream")

        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.gelu_(a)

        ops.host_store(b_d2h, b)
        ops.host_store(c_d2h, c)

    session = setup_ir(ir, [b_d2h, c_d2h])

    with pytest.raises(popart.popart_exception) as e_info:
        session.checkInplacingAmbiguity()
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_4():
    """As above failing example, but operations are now forced in order.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.gelu_(); // gelu inplace

   init -> relu_ -> gelu_ enforced.

    """
    ir = pir.Ir()

    main = ir.main_graph

    with main, pir.in_sequence():
        b_d2h = pir.d2h_stream(_IN_SHAPE, pir.dtypes.float32, name="b_stream")
        c_d2h = pir.d2h_stream(_IN_SHAPE, pir.dtypes.float32, name="c_stream")

        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.gelu_(a)

        ops.host_store(b_d2h, b)
        ops.host_store(c_d2h, c)

    session = setup_ir(ir, [b_d2h, c_d2h])

    try:
        session.checkInplacingAmbiguity()
    except popart.popart_exception as e_info:
        pytest.fail(f"Unexpected popart failure: `{e_info}`")


def test_inplacing_ambiguity_subgraph_5():
    """The classic diamond inplace graph, where a relu is performed on a and c is added to b inplace.

        a
      /   \
   relu  relu_
     |     |
     b  += c
     |
     d
    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b = ops.relu(a)
        c = ops.relu_(a)
        d = ops.add_(b, c)

    d_d2h = host_store_and_return_d2h_stream(main, d)

    session = setup_ir(ir, [d_d2h])

    with pytest.raises(popart.popart_exception) as e_info:
        session.checkInplacingAmbiguity()
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_6():
    """Same as above, but both relus are outplace, resolving the ambiguity.

       a
     /   \
     b += c
     |
     d
    """
    ir = pir.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = pir.variable(a_data, name="a")

        b = ops.relu(a)
        c = ops.relu(a)
        d = ops.add_(b, c)

    d_d2h = host_store_and_return_d2h_stream(main, d)

    session = setup_ir(ir, [d_d2h])

    try:
        session.checkInplacingAmbiguity()
    except popart.popart_exception as e_info:
        pytest.fail(f"Unexpected popart failure {e_info}")


def setup_ir(ir: pir.Ir, host_stores: List[pir.DeviceToHostStream]
             ) -> popart.InferenceSession:
    """Simple function to take an ir and create a session for.

    Args:
        ir (pir.Ir): The ir to prepare
        host_stores (List[pir.DeviceToHostStream]): The d2h streams to use in the anchors.

    Returns:
        popart.InferenceSession: The session using the ir.
    """
    arts = {}

    art_all = popart.AnchorReturnType("All")
    for s in host_stores:
        arts[s.tensor_id] = art_all

    bps = 1
    dataFlow = popart.DataFlow(bps, arts)
    ir._pb_ir.setDataFlow(dataFlow)

    ir._pb_ir.updateVertices()
    ir._pb_ir.setPatterns(
        _ir.patterns.Patterns(_ir.patterns.PatternsLevel.Minimal))

    opts = ir._pb_ir.getSessionOptions()
    # The option here should be set when creating a pir.Ir but we add them here to be explicit.
    opts.enableInplaceAmbiguityChecking = True

    session = popart.InferenceSession.fromIr(
        ir=ir._pb_ir, deviceInfo=tu.create_test_device())
    ir._pb_ir.logIr()

    return session
