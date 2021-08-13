// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PACKEDDATABLOCK_HPP
#define GUARD_NEURALNET_PACKEDDATABLOCK_HPP

#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>

namespace popart {

// 3 tensors make up a packed sequence.
// `data` is a tensor of sequences packed back to back in the first dimension.
// `offsets` and `lengths` are the offsets and lengths of these sequences in
// `data`.
struct PackedSequences {
  Tensor *data;
  Tensor *offsets;
  Tensor *lengths;
};

// The PackedDataBlockOp is an op to simplify working with sequences of packed
// data. The input is a tensor of variable length sequences, packed into the
// first dimension of a tensor. The PackedDataBlockOp handles the unpacking of
// these sequences. The user supplies to the op, a graph which is able to
// process these unpacked sequences, without having to worry about the packing
// and unpacking.
//
// PackedDataBlockOp inputs:
// 0     inData0
// 1     inOffsets0
// 2     inLengths0
// ...
// 3N    inDataN
// 3N+1  inOffsetsN
// 3N+2  inLengthsN
// 4N    outOffsets
// 4N+1  outLengths
//
// PackedDataBlockOp outputs:
// 0     outData
//
// Callback graph inputs
// 0     slicedInData0
// ...
// N     slicedInDataN
//
// Callback graph outputs:
// 0     slicedOutData
//
// For every input to the callback graph, there are 3 inputs to the
// PackedDataBlockOp. The last two inputs to the PackedDataBlockOp are the
// offsets and lengths for slicing the callback output into the output of the
// PackedDataBlockOp.
//
// The below python code demonstrates the function of the packed data block op:
//   # Input tensors
//   data, offsets, lengths, result_offsets, result_lengths = ...
//   # Input constants
//   max_tokens_per_sequence, result_size, callback_batch_size = ...
//
//   result = None
//   for i in range(len(offsets) / callback_batch_size):
//      # Collect `callback_batch_size` sequences and stack them.
//      sequences = []
//      for j in range(callback_batch_size):
//        idx = i * callback_batch_size + j
//
//        # Get sequence `idx` from data.
//        offset = offsets[idx]
//        length = lengths[idx]
//        sequence = data[offset:offset+length]
//
//        # Pad the first dimension to length, `max_tokens_per_sequence`.
//        padding = [(0,0)] * len(data.shape)
//        padding[0] = (0, max_tokens_per_sequence - sequence.shape[0])
//        sequence = np.pad(sequence, padding)
//        sequences.append(sequence)
//      np.stack(sequences)
//
//      # Call the body graph on the sequence
//      r = callback(sequences)
//
//      # Initialize the result
//      if result is None:
//        result_shape = r.shape[1:]
//        result = np.zeros([result_size] + list(result_shape))
//
//      # Pack results into result into the result
//      for j in range(callback_batch_size):
//        idx = i * callback_batch_size + j
//        result_offset = result_offsets[idx]
//        result_length = result_lengths[idx]
//        result[result_offset:result_offset+result_length] = r[j]
//
// The above python code represents this call to the PackedDataBlockOp:
//   # Input tensors
//   data, offsets, lengths, result_offsets, result_lengths = ...
//   # Input constants
//   max_tokens_per_sequence, result_size, callback_batch_size = ...
//
//   result = packedDataBlockOp([data, offset, length,
//                              result_offset, result_lengths],
//                              [max_tokens_per_sequence],
//                              result_size,
//                              callback_batch_size,
//                              callback)
//
class PackedDataBlockOp : public SubgraphOp {
public:
  PackedDataBlockOp(const OperatorIdentifier &,
                    const std::vector<int64_t> &maxSequenceLengths,
                    int64_t resultSize,
                    int64_t callbackBatchSize,
                    Graph &callback,
                    const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  InIndex subgraphInToOpInIndex(InIndex index) const override;
  InIndex opInToSubgraphInIndex(InIndex index) const override;

  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override;
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override;

  Graph &getCalledGraph() const override;
  void setCalledGraph(Graph &) override;

  // Get the number of inputs of the called graph.
  int64_t numCallbackInputs();
  // How many PackedDataBlock inputs are packed data tensors.
  int64_t numDataInputs();

  // How many calls to calledGraph need to be made.
  int64_t getCallbackIterations();

  // Return all the ops packed sequence tensors, with the corresponding offset
  // and length tensors.
  std::vector<PackedSequences> getPackedInputs();
  // Return the packed sequence output with the corresponding offset and length
  // tensors.
  PackedSequences getPackedOutput();

  // Return the `i`th packed data tensor.
  InIndex dataIndex(InIndex i) { return i * 3; }
  // Return the offsets for the `i`th packed data tensor.
  InIndex offsetsIndex(InIndex i) { return (i * 3) + 1; }
  // Return the lengths for the `i`th packed data tensor.
  InIndex lengthsIndex(InIndex i) { return (i * 3) + 2; }

  int64_t getCallbackBatchSize() { return callbackBatchSize; }
  std::vector<int64_t> getMaxSequenceLengths() { return maxSequenceLengths; }
  int64_t getMaxSequenceLength(int64_t dataIndex) {
    return maxSequenceLengths.at(dataIndex);
  }

  std::vector<TensorInfo> callbackSequenceInInfos();

private:
  const std::vector<int64_t> maxSequenceLengths;
  const int64_t resultSize;
  const int64_t callbackBatchSize;
  std::reference_wrapper<Graph> callback;
};

} // namespace popart

#endif
