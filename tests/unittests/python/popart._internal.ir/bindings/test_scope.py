# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import popart._internal.ir as _ir


def test_scope_init():
    """Test constructor."""
    _ = _ir.Scope()


def test_scope_delimiter():
    """Test delimiter()."""
    assert type(_ir.Scope.delimiter()) == str


def test_scope_empty():
    """Test empty()."""
    scope = _ir.Scope()
    assert scope.empty()
    scope = scope / "g1"
    assert not scope.empty()


def test_scope_pop():
    """Test pop()."""
    scope = _ir.Scope() / "parent"
    assert not scope.empty()
    scope.pop()
    assert scope.empty()


def test_scope_get_common_parent():
    """Test getCommonParent() and operator==, operator!=."""
    scope1 = _ir.Scope() / "base" / "sub" / "scope1"
    scope2 = _ir.Scope() / "base" / "sub" / "scope2"
    scopeCommon = _ir.Scope() / "base" / "sub"
    assert scopeCommon == scope1.getCommonParent(scope2)
    assert not scopeCommon != scope1.getCommonParent(scope2)


def test_scope_depth():
    """Test depth()."""
    scope1 = _ir.Scope() / "level1" / "level2"
    scope2 = _ir.Scope() / "level1" / "level2" / "level3"
    assert scope1.depth() == 2
    assert scope2.depth() == 3


def test_scope_str():
    """Test depth()."""
    scope = _ir.Scope() / "level1" / "level2"
    assert scope.str() == f"level1{_ir.Scope.delimiter()}level2"


def test_scope_is_sub_scope():
    """Test isSubscope()."""
    scope1 = _ir.Scope() / "level1" / "level2"
    scope2 = _ir.Scope() / "level1" / "level2" / "level3"
    assert not scope1.isSubscope(scope1)
    assert not scope2.isSubscope(scope2)
    assert not scope1.isSubscope(scope2)
    assert scope2.isSubscope(scope1)


def test_scope_get_common_parent_static():
    """Test getCommonParent_static()."""
    # TODO T42243: Add unittest for _ir.Scope.getCommonParent_static binding
    # once we have bindings for ops.
