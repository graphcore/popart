# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest


def np_normalize_image(img, offsets, scales, scale):
    expected = img.copy()
    im_shape = img.shape

    # Cast and normalize (elementwise, then broadcasted scales and offsets).
    expected = ((expected.astype(img.dtype) * scale) - offsets) * scales

    # Pad to 4 channels.
    padding = np.zeros(list(im_shape[:-1]) + [4 - im_shape[-1]])
    expected = np.c_[expected, padding]
    return expected


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.uint8])
def test_normalize_image(dtype, op_tester):
    img_shape = [2, 2, 2, 3]
    img = (np.random.uniform(0, 1, img_shape) * 255).astype(dtype)

    outDtype = np.float16 if dtype in [np.float16, np.uint8] else np.float32
    offsets = np.array([1, 2, 3], outDtype)
    scales = np.array([4, 5, 6], outDtype)
    scale = 1.0 / 255

    def init_builder(builder):
        i1 = builder.addInputTensor(img)
        i2 = builder.addInputTensor(offsets)
        i3 = builder.addInputTensor(scales)
        o = builder.aiGraphcore.normalize_image([i1, i2, i3], scale=scale)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):
        out = np_normalize_image(img, offsets, scales, scale).astype(outDtype)
        return [out]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    if dtype in [np.float16, np.uint8]:
        op_tester.rtol = 1e-2
        op_tester.atol = 1e-5
    op_tester.run(init_builder, reference, step_type="infer")
