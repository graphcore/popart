# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart


def test_constant_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == "Constant"]
    assert len(ops) > 0
    print(ops)


def test_shape_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == "Shape"]
    assert len(ops) > 0
    print(ops)


def test_constantofshape_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == "ConstantOfShape"]
    assert len(ops) > 0
    print(ops)


def test_unsupported_operations():
    opset_version = 10

    def opidToStr(opid):
        return f"{opid.type}_{opid.version}"

    unsupportedOps = popart.getUnsupportedOperations(opset_version)
    unsupportedOps = [opidToStr(i) for i in unsupportedOps]
    unsupportedOps = set(unsupportedOps)

    supportedOps = popart.getSupportedOperations(False)
    supportedOps = [opidToStr(i) for i in supportedOps]
    supportedOps = set(supportedOps)

    print(unsupportedOps & supportedOps)
    # Make sure both sets contain elements.
    assert len(unsupportedOps) > 0 and len(supportedOps) > 0
    # There should be no ops that are in both lists.
    assert len(unsupportedOps & supportedOps) == 0
