# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import itertools

import popart._internal.ir as _ir


def test_graphid_construction():
    """ Test that we can construct a popart._internal.ir.GraphId object. """
    _ = _ir.GraphId("g")


def test_graphid_operator_lt():
    """ Test the < operator. """
    for xstr, ystr in itertools.product(["g1", "g2", "y7", "z123"], repeat=2):
        x = _ir.GraphId(xstr)
        y = _ir.GraphId(ystr)

        x_le_y = x < y
        y_le_x = y < x

        # We can't violate assymetry
        assert not (x_le_y and y_le_x)

        if xstr == ystr:
            # Expect irreflexivity: neither x < y or y < x
            assert (not x_le_y) and (not y_le_x)
        else:
            # Expect totality: one of x < y or y < x
            assert x_le_y or y_le_x


def test_graphid_operator_eq_and_neq():
    """ Test the == and != operators. """

    for xstr, ystr in itertools.product(["g1", "g2", "y7", "z123"], repeat=2):
        x = _ir.GraphId(xstr)
        y = _ir.GraphId(ystr)

        if xstr == ystr:
            assert x == y
            assert not (x != y)
        else:
            assert not (x == y)
            assert x != y


def test_graphid_str():
    """ Test GraphId.str() returns the ID as a string. """
    id1 = _ir.GraphId("g1")
    assert id1.str() == "g1"
    id2 = _ir.GraphId("foobar")
    assert id2.str() == "foobar"
