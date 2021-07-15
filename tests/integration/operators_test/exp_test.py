# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from op_tester import op_tester


def test_logexp0(op_tester):
    """
    Test of both ExpInplace (priority > 0) and Exp (priority <= 0)
    """
    for inplace_priority in [-100., +100.]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            o0 = builder.aiOnnx.exp([i1])
            builder.setInplacePreferences(o0, {"ExpInplace": inplace_priority})
            o1 = builder.aiOnnx.log([o0])
            o2 = builder.aiOnnx.exp([o1])
            builder.setInplacePreferences(o2, {"ExpInplace": inplace_priority})
            o3 = builder.aiOnnx.log([o2])
            builder.addOutputTensor(o3)
            return [o3]

        def reference(ref_data):
            """
            exp(log(exp(log)))(x) = x
            """
            a = torch.tensor(d1, requires_grad=True)
            return [a * 1.0]

        op_tester.run(init_builder, reference, 'infer')


def test_exp0(op_tester):
    """
    Test of both
    1) 1 ExpInplace and 2 Exp (priority > 0) and 
    2) 3 Exp (priority <= 0)
    """
    for inplace_priority in [-100., +100.]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            # exp 1
            o1 = builder.aiOnnx.exp([i1])
            builder.setInplacePreferences(o1, {"ExpInplace": inplace_priority})
            # exp 2
            o2 = builder.aiOnnx.exp([i1])
            builder.setInplacePreferences(o2, {"ExpInplace": inplace_priority})
            # exp 3
            o3 = builder.aiOnnx.exp([i1])
            builder.setInplacePreferences(o3, {"ExpInplace": inplace_priority})

            o4 = builder.aiOnnx.sum([o1, o2, o3])
            return [o4]

        def reference(ref_data):
            """
            3*exp(in)
            """
            a = torch.tensor(3 * np.exp(d1), requires_grad=True)
            return [a]

        op_tester.run(init_builder, reference, 'infer')
