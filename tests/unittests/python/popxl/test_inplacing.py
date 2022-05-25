# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import popxl
import popxl.ops as ops
import pytest
from popxl import Tensor

_BATCH_SIZE = 6
_IN_FEATURES = 8

_IN_SHAPE = (_BATCH_SIZE, _IN_FEATURES)


def host_store_and_return_d2h_stream(g: popxl.Graph,
                                     t_: Tensor) -> popxl.DeviceToHostStream:
    """Create a host store op with the provided graph and return the stream.

    Args:
        g (popxl.Graph): The graph to use
        t_ (Tensor): The tensor to host load.

    Returns:
        popxl.DeviceToHostStream: The host stream created.
    """
    with g:
        d2h = popxl.d2h_stream(t_.shape, t_.dtype, name=t_.name + "_stream")
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
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = popxl.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = popxl.variable(w1_data, name="w1")

        m: Tensor = ops.relu_(w0)
        w0 += w1

    m_d2h = host_store_and_return_d2h_stream(main, m)

    with pytest.raises(popart.popart_exception) as e_info:
        _ = setup_ir(ir)
    assert (e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph():
    """ Check there is the same ambiguity in a called subgraph"""

    class InplaceGraph(popxl.Module):
        def __init__(self):
            pass

        def build(self, w0: Tensor, w1: Tensor):
            m: Tensor = ops.relu_(w0)
            w0 += w1
            return m, w0

    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = popxl.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = popxl.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, w0, w1)
        w0, m = ops.call(inplace_graph, w0, w1)

        m_d2h = host_store_and_return_d2h_stream(main, m)

        with pytest.raises(popart.popart_exception) as e_info:
            _ = setup_ir(ir)
        assert (
            e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph_1():
    """Same as above, but for a weight passed as a subgraph input"""

    class InplaceGraph(popxl.Module):
        def __init__(self):
            self.w1: popxl.Tensor = None

        def build(self, w0: Tensor):
            self.w1 = popxl.graph_input(w0.shape, w0.dtype, "w1")
            m: Tensor = ops.relu_(w0)
            w0 += self.w1
            return m, w0

    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        w0_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w0 = popxl.variable(w0_data, name="w0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = popxl.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, w0)
        w0, m = ops.call(inplace_graph, w0, inputs_dict={inplace_.w1: w1})

        _ = host_store_and_return_d2h_stream(main, m)

        with pytest.raises(popart.popart_exception) as e_info:
            _ = setup_ir(ir)
        assert (
            e_info.value.args[0].startswith("Inplacing ambiguity detected."))


def test_inplacing_ambiguity_subgraph_2():
    """As above but with a host load, rather than variable."""

    class InplaceGraph(popxl.Module):
        def __init__(self):
            self.w1: popxl.Tensor = None

        def build(self, x0: Tensor):
            self.w1 = popxl.graph_input(x0.shape, x0.dtype, "w1")
            m: Tensor = ops.relu_(x0)
            x0 += self.w1
            return m, x0

    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        input_ = popxl.h2d_stream(_IN_SHAPE,
                                  popxl.float32,
                                  name="input_stream")
        x0 = ops.host_load(input_, "x0")

        w1_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        w1 = popxl.variable(w1_data, name="w1")

        inplace_ = InplaceGraph()
        inplace_graph = ir.create_graph(inplace_, x0)
        x0, m = ops.call(inplace_graph, x0, inputs_dict={inplace_.w1: w1})

        _ = host_store_and_return_d2h_stream(main, m)

        with pytest.raises(popart.popart_exception) as e_info:
            _ = setup_ir(ir)
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
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        b = popxl.variable(b_data, name="b")

        d = a + 5
        d = ops.var_updates.copy_var_update_(a, b)
        e = d + 5

    _ = host_store_and_return_d2h_stream(main, e)
    _ = host_store_and_return_d2h_stream(main, e)

    with pytest.raises(popart.popart_exception) as e_info:
        _ = setup_ir(ir)
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_false_positive_2():
    """Simple example of a false positive inplacing ambiguity. Poprithms can't tell that
    relu_ is idempotent.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.relu_(); // if this were gelu_, there would be a genuine ambiguity,
    as `relu(a) == relu(relu(a))` but `gelu(relu(a)) != relu(gelu(a)) != relu(a)`.

    """
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.relu_(
            a)  #if this were gelu_, there would be a genuine ambiguity.

    _ = host_store_and_return_d2h_stream(main, b)
    _ = host_store_and_return_d2h_stream(main, c)

    with pytest.raises(popart.popart_exception) as e_info:
        _ = setup_ir(ir)
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_3():
    """As above but the second relu_ is now a gelu_ which causes a genuine ambiguity.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.gelu_(); // This is now a genuine ambiguity.
     See `test_inplacing_ambiguity_false_positive_2` for why.

    """
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        b_d2h = popxl.d2h_stream(_IN_SHAPE,
                                 popxl.dtypes.float32,
                                 name="b_stream")
        c_d2h = popxl.d2h_stream(_IN_SHAPE,
                                 popxl.dtypes.float32,
                                 name="c_stream")

        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.gelu_(a)

        ops.host_store(b_d2h, b)
        ops.host_store(c_d2h, c)

    with pytest.raises(popart.popart_exception) as e_info:
        _ = setup_ir(ir)
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_4():
    """As above failing example, but operations are now forced in order.

   a <- init();
   b <- a.relu_(); // relu inplace
   c <- a.gelu_(); // gelu inplace

   init -> relu_ -> gelu_ enforced.

    """
    ir = popxl.Ir()

    main = ir.main_graph

    with main, popxl.in_sequence():
        b_d2h = popxl.d2h_stream(_IN_SHAPE,
                                 popxl.dtypes.float32,
                                 name="b_stream")
        c_d2h = popxl.d2h_stream(_IN_SHAPE,
                                 popxl.dtypes.float32,
                                 name="c_stream")

        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b = ops.relu_(a)
        c = ops.gelu_(a)

        ops.host_store(b_d2h, b)
        ops.host_store(c_d2h, c)

    try:
        _ = setup_ir(ir)
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
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b = ops.relu(a)
        c = ops.relu_(a)
        d = ops.add_(b, c)

    _ = host_store_and_return_d2h_stream(main, d)

    with pytest.raises(popart.popart_exception) as e_info:
        _ = setup_ir(ir)
    assert e_info.value.args[0].startswith("Inplacing ambiguity detected.")


def test_inplacing_ambiguity_subgraph_6():
    """Same as above, but both relus are outplace, resolving the ambiguity.

       a
     /   \
     b += c
     |
     d
    """
    ir = popxl.Ir()

    main = ir.main_graph
    with main:
        a_data = np.random.normal(0, 0.1, _IN_SHAPE).astype(np.float32)
        a = popxl.variable(a_data, name="a")

        b = ops.relu(a)
        c = ops.relu(a)
        d = ops.add_(b, c)

    _ = host_store_and_return_d2h_stream(main, d)

    try:
        _ = setup_ir(ir)
    except popart.popart_exception as e_info:
        pytest.fail(f"Unexpected popart failure {e_info}")


def setup_ir(ir: popxl.Ir) -> popxl.Session:
    """Simple function to take an ir and create a session for.

    Args:
        ir (popxl.Ir): The ir to prepare

    Returns:
        popxl.Session: The session using the ir.
    """

    opts = ir._pb_ir.getSessionOptions()
    # The option here should be set when creating a popxl.Ir but we add them here to be explicit.
    opts.enableInplaceAmbiguityChecking = True

    session = popxl.Session(ir, "ipu_model")

    ir._pb_ir.logIr()

    return session
