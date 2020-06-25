import popart


def test_constant_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == 'Constant']
    assert len(ops) > 0
    print(ops)


def test_shape_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == 'Shape']
    assert len(ops) > 0
    print(ops)


def test_constantofshape_is_supported():
    ops = popart.getSupportedOperations(False)
    ops = [i.type for i in ops if i.type == 'ConstantOfShape']
    assert len(ops) > 0
    print(ops)
