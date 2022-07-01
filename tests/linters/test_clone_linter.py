# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from scripts.lint.linters.clone_linter import get_op_definition_start, get_closing_bracket, is_clone_signature_present, check_if_clone_is_defined
from pathlib import Path


def test_get_op_definition_start():
    """Test that we are able to capture the starting line of a Op definition."""
    test_lines = [
        "class HostLoadOp : public ExchangeBaseOp {",
        "Line with index 1, 2 and 3 will not give a hit",
        "class Foo : public Bar {", "class MultiConvOptions {",
        "class HostLoadOp : private ExchangeBaseOp {",
        "class HostLoadOp : protected ExchangeBaseOp {",
        "class HostLoadOp : private ExchangeBaseOp, private Foo {",
        "class HostLoadOp : public Foo, private ExchangeBaseOp {",
        "class HostLoadOp : public Foo, private ExchangeBaseOp, protected Baz {",
        "class ElementWiseUnaryOp : public Op {"
    ]
    result = get_op_definition_start(test_lines)
    expected = [i for i in range(len(test_lines))]
    expected.remove(1)
    expected.remove(2)
    expected.remove(3)

    assert result == expected

    test_lines = [
        "class Atan2Arg0GradOp;",  # Should not match forward declarations
        "class Atan2Arg1GradOp;",
        "",
        "class Atan2Op",
        "    : public ElementWiseNpBroadcastableBinaryWithGradOp<Atan2Arg0GradOp,",
        # Should have a match at index 5
        "                                                        Atan2Arg1GradOp> {",
        "public:",
        "  Atan2Op(const OperatorIdentifier &_opid, const Op::Settings &settings);",
        "  std::unique_ptr<Op> clone() const override;",
        "",
        "private:",
        "  bool hasLhsInplaceVariant() const final { return true; }",
        "  std::unique_ptr<Op> getLhsInplaceVariant() const final;",
        "  OperatorIdentifier getLhsOperatorIdentifier() const final;",
        "};",
        "",
        "class Atan2LhsInplaceOp",
        # Should have a match at index 17
        "    : public ElementWiseBinaryInplaceLhsOp<Atan2LhsInplaceOp> {",
        "public:",
        "  Atan2LhsInplaceOp(const Op::Settings &_settings)",
        "      : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::Atan2Inplace,",
        r"                                      _settings) {}",
        "};"
    ]

    expected = [5, 17]
    result = get_op_definition_start(test_lines)
    assert result == expected


def test_get_closing_bracket():
    """Test that we are able to get closing bracket."""
    test_lines = [
        "This is an balanced bracket {",
        "{ lorem { ipsum",
        " dolor } sit",
        "amet}",
        "}",  # Here we have the first balance
        "{ Here we add",
        " a new set of } brackets",
        "just to make the test more robust "
    ]
    result = get_closing_bracket(test_lines)
    expected = 4  # Line number of closing bracket
    assert result == expected

    test_lines = [
        "This is an unbalanced bracket {", "{ lorem ipsum", " dolor } sit",
        "amet"
    ]
    result = get_closing_bracket(test_lines)
    assert result is None


def test_is_clone_definition_present():
    """Test that the clone definition can be found."""
    test_lines = [
        "This has a definition", "just look at unique_ptr<Op> clone(), so",
        "this should return True"
    ]
    result = is_clone_signature_present(test_lines)
    assert result

    test_lines = [
        "This does not have", "The definition, so", "this should return False"
    ]
    result = is_clone_signature_present(test_lines)
    assert not result


def test_check_if_clone_is_defined(tmp_path: Path):
    """Test that the linter is working.

    Args:
        tmp_path (Path): Temporary path from pytest
    """
    # Test a successful case
    text = (
        "class HostLoadOp : public ExchangeBaseOp {\n"
        "public:\n"
        "  HostLoadOp(const OperatorIdentifier &, const Op::Settings &, TensorId sid_);\n"
        "\n"
        "  std::unique_ptr<Op> clone() const override;\n"  # 4th index
        "  void setup() final;\n"
        "\n"
        "private:\n"
        "  TensorId hostStreamTensorId {return TensorId()}\n"
        "};\n")
    tmp_success_file = tmp_path.joinpath("success.hpp")
    tmp_success_file.write_text(text)
    result = check_if_clone_is_defined(str(tmp_success_file))
    expected = 0
    assert result == expected

    # Test an Op class missing clone()
    tmp_fail_file = tmp_path.joinpath("fail.hpp")
    fail_text_list = text.split("\n")
    fail_text_list.pop(4)
    fail_text = "\n".join(fail_text_list)
    tmp_fail_file.write_text(fail_text)
    result = check_if_clone_is_defined(str(tmp_fail_file))
    expected = 1
    assert result == expected

    # Test a non-op file
    tmp_not_op_file = tmp_path.joinpath("not_an_op.hpp")
    not_op_text_list = text.split("\n")
    not_op_text_list[0] = "class Foo: Bar {"
    not_op_text = "\n".join(not_op_text_list)
    tmp_not_op_file.write_text(not_op_text)
    result = check_if_clone_is_defined(str(tmp_not_op_file))
    expected = 0

    # Test for missing newlines
    text = (
        "class SubtractArg1GradOp\n"
        "    : public ElementWiseBinaryArg1GradOp<SubtractArg1GradOp> {\n"
        "public:\n"
        "  SubtractArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);\n"
        "\n"
        "private:\n"
        "  TensorInfo forward_op_arg_info;\n"
        "};class SubtractArg1GradOp\n"
        "    : public ElementWiseBinaryArg1GradOp<SubtractArg1GradOp> {\n"
        "public:\n"
        "  SubtractArg1GradOp(const Op &, const std::vector<int64_t> &_reduction_axes);\n"
        "\n"
        "private:\n"
        "  TensorInfo forward_op_arg_info;\n"
        "};\n")
    tmp_success_file = tmp_path.joinpath("newline_missing.hpp")
    tmp_success_file.write_text(text)
    result = check_if_clone_is_defined(str(tmp_success_file))
    expected = 1
    assert result == expected
