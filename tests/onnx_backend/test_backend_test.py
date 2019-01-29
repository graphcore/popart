from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
import onnx.shape_inference
import onnx.version_converter
from typing import Optional, Text, Any, Tuple, Sequence
from onnx import NodeProto, ModelProto, TensorProto
import numpy  # type: ignore

import poponnx

# The following just executes the fake backend through the backend test
# infrastructure. Since we don't have full reference implementation of all ops
# in ONNX repo, it's impossible to produce the proper results. However, we can
# run 'checker' (that's what base Backend class does) to verify that all tests
# fed are actually well-formed ONNX models.
#
# If everything is fine, all the tests would be marked as "skipped".
#
# We don't enable report in this test because the report collection logic itself
# fails when models are mal-formed.


class Context:
    def __init__(self, s, m):
        self.session = s
        self.model = m

    def run(self, inputs):

        #print(self.session)
        # Create buffers to receive results from the execution
        anchors = self.session.initAnchorArrays()

        inputmap = {}
        i = 0

        for inp in self.model.graph.input:

            isInitializer = False
            for init in self.model.graph.initializer:
                if inp.name == init.name:
                    isInitializer = True
                    break

            if isInitializer == False:
                inputmap[str(inp.name)] = inputs[i]
                i = i + 1

        self.session.weightsFromHost()

        stepio = poponnx.PyStepIO(inputmap, anchors)
        self.session.infer(stepio)

        output_tensor = self.model.graph.output[0].name
        outputs = [anchors[output_tensor]]
        return outputs


class IpuBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(
            cls,
            model,  # type: ModelProto
            device='IPU',  # type: Text
            **kwargs  # type: Any
    ):  # type: (...) -> Optional[onnx.backend.base.BackendRep]
        super(IpuBackend, cls).prepare(model, device, **kwargs)

        # test shape inference
        model = onnx.shape_inference.infer_shapes(model)
        value_infos = {
            vi.name: vi
            for vi in itertools.chain(model.graph.value_info, model.graph.
                                      output)
        }

        # if do_enforce_test_coverage_whitelist(model):
        #     for node in model.graph.node:
        #         for i, output in enumerate(node.output):
        #             if node.op_type == 'Dropout' and i != 0:
        #                 continue
        #             assert output in value_infos
        #             tt = value_infos[output].type.tensor_type
        #             assert tt.elem_type != TensorProto.UNDEFINED
        #             for dim in tt.shape.dim:
        #                 assert dim.WhichOneof('value') == 'dim_value'

        opts = poponnx.SessionOptions()
        opts.logging = {'all': 'DEBUG'}

        anchors = {}
        for output in model.graph.output:
            anchors[output.name] = poponnx.AnchorReturnType("ALL")

        session = poponnx.Session(
            fnModel=model.SerializeToString(),
            dataFeed=poponnx.DataFlow(1, anchors),
            userOptions=opts)

        #session.setDevice(poponnx.DeviceManager().createIpuModelDevice({}))
        session.setDevice(poponnx.DeviceManager().createCpuDevice())
        session.prepareDevice()

        context = Context(session, model)

        return context

    @classmethod
    def run_node(
            cls,
            node,  # type: NodeProto
            inputs,  # type: Any
            device='IPU',  # type: Text
            outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
            **kwargs  # type: Any
    ):  # type: (...) -> Optional[Tuple[Any, ...]]
        super(IpuBackend, cls).run_node(
            node, inputs, device=device, outputs_info=outputs_info)

        raise BackendIsNotSupposedToImplementIt(
            "This is the ipu backend test that doesn't verify the results but does run the checker"
        )

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        d = Device(device)
        if d.type == DeviceType.IPU:
            return True
        return False


backend_test = onnx.backend.test.BackendTest(IpuBackend, __name__)

# Operations we do not support
backend_test.exclude('abs')
backend_test.exclude('acos')
backend_test.exclude('test_and')
backend_test.exclude('argmax')
backend_test.exclude('argmin')
backend_test.exclude('asin')
backend_test.exclude('atan')
backend_test.exclude('cast')
backend_test.exclude('ceil')
backend_test.exclude('clip')
backend_test.exclude('compress')
backend_test.exclude('constantlike')
backend_test.exclude('convtranspose')
backend_test.exclude('depthtospace')
backend_test.exclude('dynamic_slice')
backend_test.exclude('elu')
backend_test.exclude('equal')
backend_test.exclude('erf')
backend_test.exclude('expand')
backend_test.exclude('eyelike')
backend_test.exclude('flatten')
backend_test.exclude('floor')
backend_test.exclude('globalaveragepool')
backend_test.exclude('globalmaxpool')
backend_test.exclude('greater')
backend_test.exclude('gru')
backend_test.exclude('hardmax')
backend_test.exclude('hardsigmoid')
backend_test.exclude('instancenorm')
backend_test.exclude('isnan')
backend_test.exclude('leakyrelu')
backend_test.exclude('less')
backend_test.exclude('lrn')
backend_test.exclude('max')
backend_test.exclude('maxunpool')
backend_test.exclude('mean')
backend_test.exclude('min')
backend_test.exclude('mvn')
backend_test.exclude('not')
backend_test.exclude('onehot')
backend_test.exclude('or')
backend_test.exclude('pow')
backend_test.exclude('prelu')
backend_test.exclude('PReLU')
backend_test.exclude('reduce_l1')
backend_test.exclude('reduce_l2')
backend_test.exclude('reduce_log')
backend_test.exclude('reduce_max')
backend_test.exclude('reduce_mean')
backend_test.exclude('reduce_min')
backend_test.exclude('reduce_prod')
backend_test.exclude('reduce_sum')
backend_test.exclude('rnn')
backend_test.exclude('scan')
backend_test.exclude('selu')
backend_test.exclude('shape')
backend_test.exclude('sign')
backend_test.exclude('simple_rnn')
backend_test.exclude('sinh')
backend_test.exclude('size')
backend_test.exclude('softplus')
backend_test.exclude('Softplus')
backend_test.exclude('split')
backend_test.exclude('thresholdedrelu')
backend_test.exclude('tile')
backend_test.exclude('top_k')
backend_test.exclude('upsample')
backend_test.exclude('xor')
backend_test.exclude('ConvTranspose2d')
backend_test.exclude('ConvTranspose2d_no_bias')
backend_test.exclude('ELU')
backend_test.exclude('Embedding')
backend_test.exclude('Embedding_sparse')
backend_test.exclude('GLU_dim')
backend_test.exclude('GLU')
backend_test.exclude('LeakyReLU')
backend_test.exclude('LeakyReLU_with_negval')
backend_test.exclude('Linear')
backend_test.exclude('Linear_no_bias')
backend_test.exclude('SELU')

# high level models
backend_test.exclude('bvlc_alexnet')
backend_test.exclude('densenet121')
backend_test.exclude('inception_v1')
backend_test.exclude('inception_v2')
backend_test.exclude('resnet50')
backend_test.exclude('shufflenet')
backend_test.exclude('squeezenet')
backend_test.exclude('vgg19')
backend_test.exclude('zfnet512')

# The following tests cause a seg fault
backend_test.exclude('lstm')
backend_test.exclude('gemm_broadcast')

# Test that do not work for ops we have implemented

# T6601
backend_test.exclude('averagepool_1d_default')
backend_test.exclude('averagepool_3d_default')
backend_test.exclude('AvgPool3d')
backend_test.exclude('AvgPool3d_stride1_pad0_gpu_input')
backend_test.exclude('AvgPool3d_stride')
backend_test.exclude('MaxPool1d')
backend_test.exclude('MaxPool1d_stride')
backend_test.exclude('MaxPool3d')
backend_test.exclude('MaxPool3d_stride')
backend_test.exclude('MaxPool3d_stride_padding')

# T6602
backend_test.exclude('averagepool_2d_pads_count_include_pad')
backend_test.exclude('averagepool_2d_precomputed_pads_count_include_pad')

# T6603
backend_test.exclude('averagepool_2d_precomputed_same_upper')
backend_test.exclude('averagepool_2d_same_lower')
backend_test.exclude('averagepool_2d_same_upper')

# T6604
backend_test.exclude('constant')

# T6605
backend_test.exclude('gather_0')
backend_test.exclude('gather_1')
backend_test.exclude('scatter_with_axis')
backend_test.exclude('scatter_without_axis')

# T6606
backend_test.exclude('sum_one_input')

# T6607
backend_test.exclude('Conv1d_dilated')
backend_test.exclude('Conv1d_groups')
backend_test.exclude('Conv1d')
backend_test.exclude('Conv1d_pad1')
backend_test.exclude('Conv1d_pad2')
backend_test.exclude('Conv1d_stride')
backend_test.exclude('Conv2d_depthwise')
backend_test.exclude('Conv3d_dilated')
backend_test.exclude('Conv3d_dilated_strided')
backend_test.exclude('Conv3d_groups')
backend_test.exclude('Conv3d_no_bias')
backend_test.exclude('Conv3d_stride')
backend_test.exclude('Conv3d_stride_padding')
backend_test.exclude('Conv3d')

# T6608
backend_test.exclude('Conv2d_depthwise_padded')
backend_test.exclude('Conv2d_depthwise_strided')
backend_test.exclude('Conv2d_depthwise_with_multiplier')
backend_test.exclude('Conv2d_groups')
backend_test.exclude('Conv2d_groups_thnn')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()
