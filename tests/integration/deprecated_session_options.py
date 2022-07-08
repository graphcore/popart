# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import pytest


@pytest.mark.parametrize("tensor_map", [{}, {"some_tensor_id": 100}])
def test_deprecated_prefetchBufferingDepthMap_option(capfd, tensor_map):
    """
    Test deprecation of prefetchBufferingDepthMap.
    Regardless of what we set it to, the following should happen:
    * A warning is shown to indicate that the option is deprecated
    * The option gets mapped directly to it's non-deprecated alias,
      bufferingDepthMap. We test this by checking that bufferingDepthMap
      is set to the same value as prefetchBufferingDepthMap
    """
    opts = popart.SessionOptions()
    opts.prefetchBufferingDepthMap = tensor_map
    assert opts.bufferingDepthMap == tensor_map
    assert "prefetchBufferingDepthMap has been deprecated" in capfd.readouterr().err


initialDefaultPrefetchBufferingDepthValue = 111122


@pytest.mark.parametrize("value", [initialDefaultPrefetchBufferingDepthValue, 100])
def test_deprecated_defaultPrefetchBufferingDepth_option(capfd, value):
    """
    Test deprecation of defaultPrefetchBufferingDepth.
    Regardless of what we set these options to, the following should happen:
    * A warning is shown to indicate that the option is deprecated
    * The option gets mapped directly to it's non-deprecated alias,
      defaultBufferingDepth. We test this by checking that defaultBufferingDepth
      is set to the same value as defaultPrefetchBufferingDepth
    """
    opts = popart.SessionOptions()
    opts.defaultPrefetchBufferingDepth = value
    assert opts.defaultBufferingDepth == value
    assert "defaultPrefetchBufferingDepth has been deprecated" in capfd.readouterr().err
