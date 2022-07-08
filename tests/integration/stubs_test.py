# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Tests for the automatically generated stub files for popart."""
import re
from inspect import signature

import popart


def test_all_opsets_in_builder___getattr___return_annotation():
    """Test if all ONNX opset versions that are implemented in PopART are added
    to the return annotation of Builder.__getattr__().
    """
    pattern = re.compile("^AiOnnxOpset[0-9]+$")
    opsets = [prop for prop in dir(popart) if pattern.fullmatch(prop)]
    implemented_opsets = [eval(f"popart.{opset}") for opset in opsets]
    annotated_opsets = signature(popart.Builder.__getattr__).return_annotation.__args__
    for opset in implemented_opsets:
        assert opset in annotated_opsets, (
            f"Opset {opset} is implemented, but not added as return type "
            'annotation to "popart.Builder.__getattr__()".'
        )
