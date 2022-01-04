# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest


def test_import_order():
    import onnx  # type: ignore
    with pytest.raises(ImportError) as error:
        import popart  # type: ignore

    assert error.value.args[0] == (
        'It looks like onnx has already been imported. Due to an ongoing '
        'issue, popart must be imported before onnx.')
