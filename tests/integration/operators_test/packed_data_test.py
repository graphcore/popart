# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
from itertools import accumulate
import random


def gen_packed_sequences(lengths, shape, dtype=np.float32):
    sequences = []
    for length in lengths:
        sequence_shape = [length] + shape
        sequence = np.random.rand(*sequence_shape).astype(dtype)
        sequences.append(sequence)
    offsets = [0] + list(accumulate(lengths[:-1]))
    return np.concatenate(sequences), offsets


def unpack(data, offsets, lengths, sequenceLength):
    sequences = []
    for offset, length in zip(offsets, lengths):
        sequence = data[offset : offset + length]
        padding = [[0, 0] for _ in range(sequence.ndim)]
        padding[0][1] = sequenceLength - sequence.shape[0]
        sequence = np.pad(sequence, padding)
        sequences.append(sequence)
    return np.stack(sequences)


def pack(data, result, offsets, lengths):
    for i in range(data.shape[0]):
        offset = offsets[i]
        length = lengths[i]
        sequence = data[i, :length]
        result[offset : offset + length] = sequence


# `inputs`: [data1, offsets1, lengths1, ..., dataN, offsetsN, lengthsN, resultOffsets, resultLengths].
# `maxSequenceLengths`: [int] of size number_of_data_inputs.
# `destination`: The array used for the result.
# `callbackBatchSize`: Number of sequences to pass to the func.
# `func`: The function to run.
def _packed_data_block_reference(
    inputs, maxSequenceLengths, result_size, callbackBatchSize, func
):
    # get the result offsets and lengths
    assert len(inputs) > 2
    resultOffsets = inputs[-2]
    resultLengths = inputs[-1]
    inputs = inputs[:-2]

    # unpack each data input
    data_inputs = []
    assert len(inputs) % 3 == 0
    input_count = len(inputs) // 3
    for i in range(input_count):
        data = inputs[i * 3]
        offsets = inputs[(i * 3) + 1]
        lengths = inputs[(i * 3) + 2]
        maxSequenceLength = maxSequenceLengths[i]

        d = unpack(data, offsets, lengths, maxSequenceLength)
        data_inputs.append(d)

    nSequences = len(data_inputs[0])

    results = []
    for i in range(nSequences // callbackBatchSize):
        ins = []
        for di in data_inputs:
            ins.append(
                di[
                    (i * callbackBatchSize) : (i * callbackBatchSize)
                    + callbackBatchSize
                ]
            )
        r = func(*ins)
        results.append(r)

    destination_shape = [result_size] + list(results[0].shape)[2:]
    destination = np.zeros(destination_shape).astype(results[0].dtype)

    for i in range(len(results)):
        for innerSequence in range(callbackBatchSize):
            idx = i * callbackBatchSize + innerSequence

            offset = resultOffsets[idx]
            length = resultLengths[idx]
            destination[offset : offset + length] = results[i][innerSequence][:length]

    return destination


# test that the reference function is working as expected.
@pytest.mark.parametrize("callbackBatchSize", [1, 2, 3])
def test_packed_data_block_reference(callbackBatchSize):
    sequenceLengths = [3, 5, 7, 4, 6, 2]
    data, sequenceOffsets = gen_packed_sequences(sequenceLengths, [5])

    maxSequenceLength = 10

    def unpacked_ref(data):
        dt = np.transpose(data, [0, 2, 1])
        mm = np.matmul(data, dt)
        return mm

    def packed_ref(data):
        calls_to_func = 0

        def func(d):
            nonlocal calls_to_func
            calls_to_func += 1

            dt = np.transpose(d, [0, 2, 1])
            return np.matmul(d, dt)

        result = _packed_data_block_reference(
            [data, sequenceOffsets, sequenceLengths, sequenceOffsets, sequenceLengths],
            [maxSequenceLength],
            data.shape[0],
            callbackBatchSize,
            func,
        )

        # Check how many times `func` was called.
        nSequences = len(sequenceLengths)
        assert calls_to_func == nSequences // callbackBatchSize

        return result

    d = unpack(data, sequenceOffsets, sequenceLengths, maxSequenceLength)
    unpacked_result = unpacked_ref(d)

    packed_result = packed_ref(data)
    packed_result = unpack(
        packed_result, sequenceOffsets, sequenceLengths, maxSequenceLength
    )

    assert unpacked_result.shape == packed_result.shape
    assert np.array_equal(packed_result, unpacked_result)


@pytest.mark.parametrize("callbackBatchSize", [1, 2, 3])
def test_packeddatablockop(op_tester, callbackBatchSize):
    np.random.seed(0)

    sequenceLengths = [3, 5, 7, 4, 6, 2]
    data, sequenceOffsets = gen_packed_sequences(sequenceLengths, [5])
    data = (data * 9 + 1).astype(np.uint32).astype(np.float32)

    sequenceLengths = np.array(sequenceLengths).astype(np.uint32)
    sequenceOffsets = np.array(sequenceOffsets).astype(np.uint32)

    maxSequenceLength = 10

    def init_builder(builder):
        dataId = builder.addInputTensor(data, "data")
        sequenceLengthsId = builder.addInputTensor(sequenceLengths, "lengths")
        sequenceOffsetsId = builder.addInputTensor(sequenceOffsets, "offsets")

        subgraph_builder = builder.createSubgraphBuilder()

        sgi0 = subgraph_builder.addUntypedInputTensor()

        dt = subgraph_builder.aiOnnx.transpose([sgi0], [0, 2, 1])
        out = subgraph_builder.aiOnnx.matmul([sgi0, dt])

        subgraph_builder.addOutputTensor(out)

        out = builder.aiGraphcore.packedDataBlock(
            [
                dataId,
                sequenceOffsetsId,
                sequenceLengthsId,
                sequenceOffsetsId,
                sequenceLengthsId,
            ],
            [maxSequenceLength],
            data.shape[0],
            callbackBatchSize,
            subgraph_builder,
        )

        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        d = unpack(data, sequenceOffsets, sequenceLengths, maxSequenceLength)
        dt = np.transpose(d, [0, 2, 1])
        mm = np.matmul(d, dt)
        result = np.zeros([27, 10]).astype(np.float32)
        pack(mm, result, sequenceOffsets, sequenceLengths)
        return [result]

    op_tester.patterns.enablePattern("PackedDataBlock", True)
    op_tester.run(init_builder, reference, "infer")


def test_bertlike_attention(op_tester):
    random.seed(0)

    # These are the attributes in the bert model.
    hidden_size = 1024
    micro_batch_size = 8
    sequence_length = 384
    attention_heads = 16
    qkv_length = hidden_size // attention_heads

    # These are attributes specific to using packed data
    max_tokens_per_sequence = sequence_length

    lengths = [
        random.randint(2, max_tokens_per_sequence) for i in range(micro_batch_size)
    ]
    packed_data, offsets = gen_packed_sequences(
        lengths, [attention_heads * qkv_length * 3]
    )
    lengths = np.array(lengths).astype(np.uint32)
    offsets = np.array(offsets).astype(np.uint32)

    # Add padding after the sequences to change the size to the data found in bert:
    # [micro_batch_size * sequence_length, attention_heads * qkv_length * 3]
    packed_data = np.pad(
        packed_data,
        [(0, (micro_batch_size * sequence_length) - packed_data.shape[0]), (0, 0)],
    )

    def init_builder(builder):
        dataId = builder.addInputTensor(packed_data, "data")
        sequenceLengthsId = builder.addInputTensor(lengths, "lengths")
        sequenceOffsetsId = builder.addInputTensor(offsets, "offsets")

        subgraph_builder = builder.createSubgraphBuilder()

        qkv = subgraph_builder.addUntypedInputTensor()
        q, k, v = subgraph_builder.aiOnnx.split([qkv], 3, axis=2)

        def extract_head(t, perm):
            comb_shape = [sequence_length, attention_heads, qkv_length]
            t = subgraph_builder.reshape_const(subgraph_builder.aiOnnx, [t], comb_shape)
            return subgraph_builder.aiOnnx.transpose([t], perm)

        q = extract_head(q, [1, 0, 2])
        kt = extract_head(k, [1, 2, 0])
        v = extract_head(v, [1, 0, 2])

        x = subgraph_builder.aiOnnx.matmul([q, kt])
        c = np.array([1 / np.sqrt(qkv_length)]).astype(np.float32)
        c = subgraph_builder.aiOnnx.constant(c)
        x = subgraph_builder.aiOnnx.mul([x, c])
        # Disabling softmax as the result differs
        # x = subgraph_builder.aiOnnx.softmax([x])

        x = subgraph_builder.aiOnnx.matmul([x, v])
        x = subgraph_builder.aiOnnx.transpose([x], [1, 0, 2])
        x = subgraph_builder.reshape_const(
            subgraph_builder.aiOnnx, [x], [1, sequence_length, hidden_size]
        )
        out = x

        subgraph_builder.addOutputTensor(out)

        out = builder.aiGraphcore.packedDataBlock(
            [
                dataId,
                sequenceOffsetsId,
                sequenceLengthsId,
                sequenceOffsetsId,
                sequenceLengthsId,
            ],
            [max_tokens_per_sequence],
            packed_data.shape[0],
            1,
            subgraph_builder,
        )

        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        d = unpack(packed_data, offsets, lengths, max_tokens_per_sequence)

        # This is the shape of the data going into the bert attention layer
        # (micro_batch_size *sequence_length, attention_heads * qkv_length * 3)
        d = np.reshape(d, packed_data.shape)
        q, k, v = np.split(d, 3, axis=1)

        def extract_head(t, perm):
            comb_shape = [
                micro_batch_size,
                sequence_length,
                attention_heads,
                qkv_length,
            ]
            t = np.reshape(t, comb_shape)
            return np.transpose(t, perm)

        q = extract_head(q, [0, 2, 1, 3])
        kt = extract_head(k, [0, 2, 3, 1])
        v = extract_head(v, [0, 2, 1, 3])

        x = np.matmul(q, kt)
        x = x * (1 / np.sqrt(qkv_length))
        # Disabling softmax as the result differs
        # x = scipy.special.softmax(x)

        x = np.matmul(x, v)
        x = np.transpose(x, [0, 2, 1, 3])
        x = np.reshape(x, [micro_batch_size, sequence_length, hidden_size])

        result = np.zeros([micro_batch_size * sequence_length, hidden_size]).astype(
            np.float32
        )
        pack(x, result, offsets, lengths)
        return [result]

    op_tester.patterns.enablePattern("PackedDataBlock", True)
    op_tester.run(init_builder, reference, "infer")
