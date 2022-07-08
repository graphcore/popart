# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest


def test_import_order():
    import onnx  # pylint: disable=unused-import

    with pytest.raises(ImportError) as error:
        import popart  # pylint: disable=unused-import

    assert error.value.args[0] == (
        "It looks like onnx has already been imported. Due to an ongoing "
        "issue, popart must be imported before onnx."
    )
