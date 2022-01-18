// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTM_HPP
#define GUARD_NEURALNET_LSTM_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/lstmutil.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

class LSTMOp : public Op {
public:
  LSTMOp(const OperatorIdentifier &_opid,
         nonstd::optional<int64_t> hidden_size,
         ActivationFunction activation,
         ActivationFunction recurrent_activation,
         const Op::Settings &settings_,
         const nonstd::optional<float> available_memory_proportion_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  unsigned getNumChannels() const;

  int64_t getMaxSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  int64_t getNumDirections() const;
  int64_t getHiddenSize() const;
  nonstd::optional<float> getAvailableMemoryProportion() const;

  bool hasBiasInput() const;
  bool hasInitialHInput() const;
  bool hasInitialCInput() const;
  bool hasSeqLenInput() const;
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

  view::Regions aliases(InIndex, OutIndex) const final;
  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  ActivationFunction getActivation() const { return activation; }
  ActivationFunction getRecurrentActivation() const {
    return recurrent_activation;
  }

private:
  void maybeCreatePassThroughOutput(const TensorId &new_id,
                                    OutIndex pass_through_index,
                                    const TensorInfo &out_info);
  // Intermediate results that are retained in the forward pass of training for
  // use in the backward pass.
  int64_t getNumIntermediates() const;
  void trySetOutInfo(OutIndex, const TensorInfo &);

  nonstd::optional<int64_t> hidden_size_attribute;

  ActivationFunction activation;
  ActivationFunction recurrent_activation;

  nonstd::optional<float> available_memory_proportion;
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
  static InIndex getSequenceLensInIndex() { return 8; }

  static InIndex getCellStateOutputGradInIndex() { return 9; }
  static InIndex getHiddenStateOutputGradInIndex() { return 10; }
  static InIndex getOutputGradInIndex() { return 11; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceOutIndex() { return 2; }
  static OutIndex getBiasOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }
  static OutIndex getInitialCOutIndex() { return 5; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  const LSTMOp &forward_op;

  // This is usually a static variable in the method `gradInputInfo`, but for
  // this op `inInfo` can change depending on the result of
  // `getForwardOp()->hasSeqLenInput()`.
  const std::vector<GradInOutMapper> inInfo;
  // This static method allows us to calculate `inInfo` during member
  // initialization, and then `inInfo` may be marked `const`.
  static std::vector<GradInOutMapper> gradInputInfo(const LSTMOp &forwardOp);
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
               const Op::Settings &,
               const nonstd::optional<float> available_memory_proportion_ =
                   nonstd::optional_lite::nullopt);

  PopartLSTMOp(const OperatorIdentifier &,
               bool outputFullSequence_,
               ActivationFunction activation,
               ActivationFunction recurrent_activation,
               const Op::Settings &,
               const nonstd::optional<float> available_memory_proportion_ =
                   nonstd::optional_lite::nullopt);

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
  static InIndex getSequenceLensInIndex() { return 4; }

  static OutIndex getOutputOutIndex() { return 0; }
  static OutIndex getCellStateOutIndex() { return 1; }
  static OutIndex getIntermediatesOutIndex() { return 2; }

  bool hasSeqLenInput() const;

  int64_t getMaxSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;
  int64_t getHiddenSize() const;

  // Intermediate results that are retained in the forward pass of training for
  // use in the backward pass.
  int64_t getNumIntermediates() const;

  nonstd::optional<float> getAvailableMemoryProportion() const;

  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

  ActivationFunction getActivation() const { return activation; }
  ActivationFunction getRecurrentActivation() const {
    return recurrent_activation;
  }

  const bool outputFullSequence;

private:
  const ActivationFunction activation;
  const ActivationFunction recurrent_activation;

  nonstd::optional<float> available_memory_proportion;
};

class PopartLSTMGradOp : public Op {
public:
  PopartLSTMGradOp(const PopartLSTMOp &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  const PopartLSTMOp &getForwardOp() const;

  std::set<InIndex> optionalInputs() const final;

  int64_t getInputSize() const;
  int64_t getMaxSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getHiddenSize() const;

  static InIndex getInitialStateInIndex() { return 0; }
  static InIndex getIntermediatesInIndex() { return 1; }
  static InIndex getWeightsInIndex() { return 2; }
  static InIndex getBiasesInIndex() { return 3; }
  static InIndex getSequenceLensInIndex() { return 4; }
  static InIndex getInputInIndex() { return 5; }

  static InIndex getFwdOutputInIndex() { return 6; }
  static InIndex getFwdOutputGradInIndex() { return 7; }
  static InIndex getFwdCellStateGradInIndex() { return 8; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getWeightsOutIndex() { return 1; }
  static OutIndex getBiasesOutIndex() { return 2; }
  static OutIndex getInitialStateOutIndex() { return 3; }

  ActivationFunction getActivation() const { return activation; }
  ActivationFunction getRecurrentActivation() const {
    return recurrent_activation;
  }

  const bool outputFullSequence;

private:
  const TensorId forwardCellStateGradId;
  const ActivationFunction activation;
  const ActivationFunction recurrent_activation;
};

} // namespace popart

#endif
