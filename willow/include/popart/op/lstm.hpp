// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTM_HPP
#define GUARD_NEURALNET_LSTM_HPP

#include <popart/op.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

class LSTMOp : public Op {
public:
  LSTMOp(const OperatorIdentifier &_opid,
         nonstd::optional<int64_t> hidden_size,
         const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  unsigned getNumChannels() const;

  int64_t getSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  int64_t getNumDirections() const;
  int64_t getHiddenSize() const;

  bool hasBiasInput() const;
  bool hasInitialHInput() const;
  bool hasInitialCInput() const;
  bool hasOutput(OutIndex) const;

  std::set<InIndex> optionalInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isTraining() const;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getRecurrenceInIndex() { return 2; }
  static InIndex getBiasInIndex() { return 3; }
  static InIndex getSequenceLensInIndex() { return 4; }
  static InIndex getInitialHInIndex() { return 5; }
  static InIndex getInitialCInIndex() { return 6; }
  static InIndex getPeepholeInIndex() { return 7; }

  static OutIndex getOutputOutIndex() { return 0; }
  static OutIndex getHiddenStateOutIndex() { return 1; }
  static OutIndex getCellStateOutIndex() { return 2; }

  static OutIndex getInitStateOutputPassThroughIndex() { return 3; }
  static OutIndex getInitStateCellStatePassThroughIndex() { return 4; }
  static OutIndex getIntermediatesPassThroughIndex() { return 5; }
  static OutIndex getInputWeightsPassThroughIndex() { return 6; }
  static OutIndex getOutputWeightsPassThroughIndex() { return 7; }
  static OutIndex getBiasesPassThroughIndex() { return 8; }
  static OutIndex getInputPassThroughIndex() { return 9; }
  static OutIndex getOutputPassThroughIndex() { return 10; }

  // T7504
  bool isOutlineable() const override { return false; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

private:
  void createPassThroughOutput(const TensorId &new_id,
                               OutIndex pass_through_index,
                               const TensorInfo &out_info);
  static int getNumIntermediates() { return 6; }
  void trySetOutInfo(OutIndex, const TensorInfo &);

  nonstd::optional<int64_t> hidden_size_attribute;
};

class LSTMGradOp : public Op {
public:
  LSTMGradOp(const LSTMOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  const LSTMOp &getForwardOp() const;

  bool hasCellStateGradInput() const;
  bool hasHiddenStateGradInput() const;

  std::set<InIndex> optionalInputs() const final;

  static InIndex getInitStateOutputInIndex() { return 0; }
  static InIndex getInitStateCellStateInIndex() { return 1; }
  static InIndex getIntermediatesInIndex() { return 2; }
  static InIndex getInputWeightsInIndex() { return 3; }
  static InIndex getOutputWeightsInIndex() { return 4; }
  static InIndex getBiasesInIndex() { return 5; }
  static InIndex getInputInIndex() { return 6; }
  static InIndex getOutputInIndex() { return 7; }

  static InIndex getCellStateOutputGradInIndex() { return 8; }
  static InIndex getHiddenStateOutputGradInIndex() { return 9; }
  static InIndex getOutputGradInIndex() { return 10; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceOutIndex() { return 2; }
  static OutIndex getBiasOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }
  static OutIndex getInitialCOutIndex() { return 5; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  const LSTMOp &forward_op;
};

// LSTM op that more closely resembles the poplar lstm.
// Inputs:
//   X: The input sequence of shape,
//      [sequence_length, batch_size, input_size].
//   weights: The input and recurrence weights, of shape,
//            [4, input_size + hidden_size, hidden_size].
//   biases: The bias for the input gate, of shape,
//           [4, hidden_size].
//
// Outputs:
//   Y: The output values. If outputFullSequence is true, this
//      has the shape, [sequence_length, batch_size, hidden_size]
//      otherwise it is [batch_size, hidden_size]
//   cell_state: The last output value of the cell, of shape,
//               [batch_size, hidden_size]
//
// Attributes:
//   outputFullSequence: Whether to output the full sequence or
//                       just the final output.
class PopartLSTMOp : public Op {
public:
  PopartLSTMOp(const OperatorIdentifier &,
               bool outputFullSequence_,
               const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool hasBiasesInput() const;
  std::set<InIndex> optionalInputs() const final;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getWeightsInIndex() { return 1; }
  static InIndex getBiasesInIndex() { return 2; }
  static InIndex getInitialStateInIndex() { return 3; }

  static OutIndex getOutputOutIndex() { return 0; }
  static OutIndex getCellStateOutIndex() { return 1; }
  static OutIndex getIntermediatesOutIndex() { return 2; }

  int64_t getSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  int64_t getHiddenSize() const;

  static int64_t getNumIntermediates() { return 6; }

  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

  const bool outputFullSequence;
};

class PopartLSTMGradOp : public Op {
public:
  PopartLSTMGradOp(const PopartLSTMOp &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  std::set<InIndex> optionalInputs() const final;

  int64_t getInputSize() const;
  int64_t getSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getHiddenSize() const;

  static InIndex getInitialStateInIndex() { return 0; }
  static InIndex getIntermediatesInIndex() { return 1; }
  static InIndex getWeightsInIndex() { return 2; }
  static InIndex getBiasesInIndex() { return 3; }
  static InIndex getInputInIndex() { return 4; }
  static InIndex getFwdOutputInIndex() { return 5; }
  static InIndex getFwdOutputGradInIndex() { return 6; }
  static InIndex getFwdCellStateGradInIndex() { return 7; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getBiasesOutIndex() { return 2; }
  static OutIndex getInitialStateOutIndex() { return 3; }

  const bool outputFullSequence;

private:
  const TensorId forwardCellStateGradId;
};

} // namespace popart

#endif
