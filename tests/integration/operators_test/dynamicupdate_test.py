# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from op_tester import op_tester
from unittest.mock import Mock


# Test a chain of non-overlapping dynamic updates
# init -> U0 -> out0 -> U1 -> out1 -> U2 -> out2
#         ^             ^             ^
#         |             |             |
#         tensor0       tensor1       tensor2
# where tensor0, tensor1 and tensor2 are non-overlapping subregions
# of the out tensor
def test_dynamicupdate(op_tester):
    data0 = np.random.rand(5, 4, 7).astype(np.float32)
    data1 = np.random.rand(5, 4, 7).astype(np.float32)
    data2 = np.random.rand(5, 4, 7).astype(np.float32)
    axes = [1]
    sizes = [4]

    def init_builder(builder):
        tensor0 = builder.addInputTensor(data0)
        tensor1 = builder.addInputTensor(data1)
        tensor2 = builder.addInputTensor(data2)
        tensors = [tensor0, tensor1, tensor2]
        result = []
        out = builder.aiGraphcore.init([5, 12, 7], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")

        assert builder.getTensorShape(out) == [5, 12, 7]

        for sliceid in range(3):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicupdate(
                [out, index, tensors[sliceid]],
                axes=axes,
                sizes=sizes,
                noOverlap=True)

            assert builder.getTensorShape(out) == [5, 12, 7]

            builder.addOutputTensor(out)
        result.append(out)
        return result

    def reference(ref_data):
        result = []
        result.append(np.concatenate((data0, data1, data2), axis=1))
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Test dynamic update for multiple dimensions that don't divide the input evenly
def test_dynamicupdate_multi_dim(op_tester):
    data = np.random.rand(5, 12, 7).astype(np.float32)
    data_update = np.random.rand(3, 4, 5).astype(np.float32)
    axes = [0, 1, 2]
    indices = [1, 3, 2]
    sizes = data_update.shape

    def init_builder(builder):
        t_update = builder.addInputTensor(data_update)
        result = []
        out = builder.addInputTensor(data)

        assert builder.getTensorShape(out) == list(data.shape)

        index = builder.addInputTensor(np.asarray(indices, np.uint32))
        out = builder.aiGraphcore.dynamicupdate([out, index, t_update],
                                                axes=axes,
                                                sizes=sizes,
                                                noOverlap=True)

        assert builder.getTensorShape(out) == list(data.shape)

        builder.addOutputTensor(out)
        result.append(out)
        return result

    def reference(_):
        result = []
        data[indices[0]:(indices[0] + sizes[0]), indices[1]:(
            indices[1] + sizes[1]), indices[2]:(indices[2] +
                                                sizes[2])] = data_update
        result.append(data)
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Test training of non-overlapping dynamic updates
def test_dynamicupdate_training(op_tester):
    data0 = np.random.rand(5, 4, 7).astype(np.float32)
    data1 = np.random.rand(5, 4, 7).astype(np.float32)
    data2 = np.random.rand(5, 4, 7).astype(np.float32)
    axes = [1]
    sizes = [4]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensor2 = builder.addInitializedInputTensor(data2)
        tensors = [tensor0, tensor1, tensor2]
        result = []
        out = builder.aiGraphcore.init([5, 12, 7], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")
        for sliceid in range(3):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicupdate(
                [out, index, tensors[sliceid]],
                axes=axes,
                sizes=sizes,
                noOverlap=True)
        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0, 1, 2], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
            popart.reservedGradientPrefix() + tensor2,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)
        tensor2 = torch.tensor(data2, requires_grad=True)

        outputs = []
        result = []
        out = torch.cat((tensor0, tensor1, tensor2), dim=1)
        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        result = [
            sum,
            torch.tensor(d__o), tensor0.grad, tensor1.grad, tensor2.grad
        ] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic updates are incorrect if noOverlap
# is set to True while the sliced regions overlap
def test_dynamicupdate_overlap_wrong(op_tester):
    data0 = np.random.rand(5).astype(np.float32)
    data1 = np.random.rand(5).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensors = [tensor0, tensor1]
        result = []
        out = builder.aiGraphcore.init([10], popart.DataType.FLOAT,
                                       popart.InitType.Zero, "test_init")
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            scaled = builder.aiGraphcore.scale([tensors[sliceid]],
                                               float(1 + sliceid))
            out = builder.aiGraphcore.dynamicupdate([out, index, scaled],
                                                    axes=axes,
                                                    sizes=sizes,
                                                    noOverlap=True)
        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)

        outputs = []
        result = []

        out = torch.zeros(10)

        out[0:5] = tensor0 * 1.0
        out[4:9] = tensor1 * 2.0

        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: We have to adjust the value here to make the comparison equal,
        # but dynamicupdate with noOverlap=True gives a wrong gradient result
        # due to overlapping updates
        tensor0.grad[4] += 1.0

        result = [sum, torch.tensor(d__o), tensor0.grad, tensor1.grad] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic updates are correct if noOverlap is
# set to False while the sliced regions overlap
def test_dynamicupdate_overlap_correct(op_tester):
    data0 = np.random.rand(5).astype(np.float32)
    data1 = np.random.rand(5).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensors = [tensor0, tensor1]
        result = []
        out = builder.aiGraphcore.init([10], popart.DataType.FLOAT,
                                       popart.InitType.Zero, "test_init")
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            scaled = builder.aiGraphcore.scale([tensors[sliceid]],
                                               float(1 + sliceid))

            out = builder.aiGraphcore.dynamicupdate([out, index, scaled],
                                                    axes=axes,
                                                    sizes=sizes,
                                                    noOverlap=False)

        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)

        outputs = []
        result = []

        out = torch.zeros(10)

        out[0:5] = tensor0 * 1.0
        out[4:9] = tensor1 * 2.0

        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: Comparison equal with noOverlap=False handles overlapping
        # updates correctly (at higher computational cost).
        # No correction needed.
        tensor0.grad[4] += 0.0

        result = [sum, torch.tensor(d__o), tensor0.grad, tensor1.grad] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_dynamicupdate_shape():
    """ A test taken from the deepvoice model, that tests the dynamicupdate
output shape is correctly inferred inside a call op. Previously this would fail
as the shape inference for the dynamicupdate, and thus the subgraph would not run.
"""

    def get_test_conf():
        conf = Mock()
        conf.samples_per_device_for_inference = 2
        conf.decoder_channels = 256
        conf.attention_hidden_size = 256
        conf.inference_look_ahead = 3
        conf.max_text_sequence_length = 80
        return conf

    def get_subbuilder(builder, conf):

        subbuilder = builder.createSubgraphBuilder()

        h_q_shape = popart.TensorInfo(
            "FLOAT",
            [conf.samples_per_device_for_inference, conf.decoder_channels, 1])
        query_positional_encoding_shape = popart.TensorInfo(
            "FLOAT",
            [conf.samples_per_device_for_inference, conf.decoder_channels, 1])
        query_projection_weights_shape = popart.TensorInfo(
            "FLOAT", [conf.attention_hidden_size, conf.decoder_channels])
        Q_k_shape = popart.TensorInfo("FLOAT", [
            conf.samples_per_device_for_inference, conf.attention_hidden_size,
            conf.max_text_sequence_length
        ])
        Q_v_shape = popart.TensorInfo("FLOAT", [
            conf.samples_per_device_for_inference, conf.attention_hidden_size,
            conf.max_text_sequence_length
        ])
        num_time_steps_shape = popart.TensorInfo("FLOAT", [1])
        context_vec_projections_weights_shape = popart.TensorInfo(
            "FLOAT", [conf.decoder_channels, conf.attention_hidden_size])

        h_q = subbuilder.addInputTensor(h_q_shape)
        query_positional_encoding = subbuilder.addInputTensor(
            query_positional_encoding_shape)
        query_projection_weights = subbuilder.addInputTensor(
            query_projection_weights_shape)
        Q_k = subbuilder.addInputTensor(Q_k_shape)
        Q_v = subbuilder.addInputTensor(Q_v_shape)
        num_time_steps = subbuilder.addInputTensor(num_time_steps_shape)
        context_vec_projections_weights = subbuilder.addInputTensor(
            context_vec_projections_weights_shape)

        # forced monotonic attention elements
        large_negative_tensor = subbuilder.addInputTensor(
            popart.TensorInfo("FLOAT", [1, 1, conf.max_text_sequence_length]))
        zeros_mask = subbuilder.addInputTensor(
            popart.TensorInfo("FLOAT", [1, 1, conf.inference_look_ahead]))
        last_attended = [
            subbuilder.addInputTensor(popart.TensorInfo("INT32", [1]))
            for _ in range(conf.samples_per_device_for_inference)
        ]

        # add positional encoding to queries
        h_q = subbuilder.aiOnnx.add([h_q, query_positional_encoding])
        Q_q = subbuilder.aiOnnx.matmul([query_projection_weights, h_q])

        # transposing Q_q
        Q_q_t = subbuilder.aiOnnx.transpose(
            [Q_q], perm=[0, 2, 1])  # 1 X attention_hidden_size

        # getting transformed query key dot products (1 X Tk)
        attention_scores = subbuilder.aiOnnx.matmul([Q_q_t, Q_k])

        # forced monotonic attention
        attention_scores_split = subbuilder.aiOnnx.split(
            [attention_scores],
            num_outputs=conf.samples_per_device_for_inference,
            axis=0)
        for sample_ind in range(conf.samples_per_device_for_inference):
            update_mask = subbuilder.aiGraphcore.dynamicupdate(
                [large_negative_tensor, last_attended[sample_ind], zeros_mask],
                axes=[2],
                sizes=[conf.inference_look_ahead],
                noOverlap=True)
            attention_scores_split[sample_ind] = subbuilder.aiOnnx.add(
                [attention_scores_split[sample_ind], update_mask])
        attention_scores = subbuilder.aiOnnx.concat(attention_scores_split,
                                                    axis=0)

        attention_scores = subbuilder.aiOnnx.softmax([attention_scores],
                                                     axis=2)

        last_attended = subbuilder.aiOnnx.argmax([attention_scores], axis=2)

        attention_scores = subbuilder.aiOnnx.transpose([attention_scores],
                                                       perm=[0, 2,
                                                             1])  # (Tk X 1)

        # getting weighted average of value vectors to get context vectors
        context_vector = subbuilder.aiOnnx.matmul(
            [Q_v, attention_scores])  # (v X 1)

        # dividing by sqrt of num-steps
        context_vector = subbuilder.aiOnnx.div(
            [context_vector,
             subbuilder.aiOnnx.sqrt([num_time_steps])])

        context_vector = subbuilder.aiOnnx.matmul(
            [context_vec_projections_weights, context_vector])

        context_vector = subbuilder.aiOnnx.relu([context_vector])

        subbuilder.addOutputTensor(context_vector)
        subbuilder.addOutputTensor(attention_scores)
        subbuilder.addOutputTensor(last_attended)

        return subbuilder

    builder = popart.Builder()
    conf = get_test_conf()
    subbuilder = get_subbuilder(builder, conf)

    h_q = builder.addInputTensor(
        popart.TensorInfo(
            "FLOAT",
            [conf.samples_per_device_for_inference, conf.decoder_channels, 1]))
    query_positional_encoding = builder.addInputTensor(
        popart.TensorInfo(
            "FLOAT",
            [conf.samples_per_device_for_inference, conf.decoder_channels, 1]))
    query_projection_weights = builder.addInputTensor(
        popart.TensorInfo("FLOAT",
                          [conf.attention_hidden_size, conf.decoder_channels]))
    Q_k = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [
            conf.samples_per_device_for_inference, conf.attention_hidden_size,
            conf.max_text_sequence_length
        ]))
    Q_v = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [
            conf.samples_per_device_for_inference, conf.attention_hidden_size,
            conf.max_text_sequence_length
        ]))
    num_time_steps = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    context_vec_projections_weights = builder.addInputTensor(
        popart.TensorInfo("FLOAT",
                          [conf.decoder_channels, conf.attention_hidden_size]))
    large_negative_tensor = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [1, 1, conf.max_text_sequence_length]))
    zeros_mask = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [1, 1, conf.inference_look_ahead]))
    last_attended = [
        builder.addInputTensor(popart.TensorInfo("INT32", [1]))
        for _ in range(conf.samples_per_device_for_inference)
    ]

    context_vector, attention_scores, last_attended = builder.aiGraphcore.call(
        [
            h_q, query_positional_encoding, query_projection_weights, Q_k, Q_v,
            num_time_steps, context_vec_projections_weights,
            large_negative_tensor, zeros_mask
        ] + last_attended,
        3,
        callee=subbuilder)

    print(builder.getTensorShape(context_vector))
    print(builder.getTensorShape(attention_scores))
    print(builder.getTensorShape(last_attended))
    assert (builder.getTensorShape(context_vector) == [
        conf.samples_per_device_for_inference, conf.decoder_channels, 1
    ])
    assert (builder.getTensorShape(attention_scores) == [
        conf.samples_per_device_for_inference, conf.max_text_sequence_length, 1
    ])
    assert (builder.getTensorShape(last_attended) == [
        conf.samples_per_device_for_inference, 1, 1
    ])
