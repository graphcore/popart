# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# Test for basic importing


def test_import():
    # the core library
    import popart

    assert "SGD" in dir(popart)
