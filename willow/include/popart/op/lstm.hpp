// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTM_HPP
#define GUARD_NEURALNET_LSTM_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/lstmutil.hpp>
#include <popart/op/rnnbase.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

/**
 * This op applies a single-layer LSTM with a non-linearity to a batch of
 * input sequences. The op follows the ONNX specification described in
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
 */
class LSTMOp : public BaseOnnxRNNOp {
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
  int64_t getNumDirections() const;

  nonstd::optional<float> getAvailableMemoryProportion() const;

  bool hasInitialCInput() const;
  bool hasOutput(OutIndex) const;

  std::set<InIndex> optionalInputs() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isTraining() const;

  // inputs 0-5 defined in BaseOnnxRNNOp
  static InIndex getInitialCInIndex() { return 6; }
  static InIndex getPeepholeInIndex() { return 7; }

  // outputs 0-1 defined in BaseOnnxRNNOp
  static OutIndex getLastCellStateOutIndex() { return 2; }

  // set to 0 if not provided by user
  static OutIndex getInitialHPassThroughIndex() { return 3; }
  // set to 0 if not provided by user
  static OutIndex getInitialCPassThroughIndex() { return 4; }
  // intermediate values needed to calculate the grad
  static OutIndex getIntermediatesPassThroughIndex() { return 5; }
  // restructured input weights to be used with poplar implementation
  static OutIndex getInputWeightsPassThroughIndex() { return 6; }
  // restructured recurrence weights to be used with poplar implementation
  static OutIndex getRecurrenceWeightsPassThroughIndex() { return 7; }
  // restructured biases to be used with poplar implementation
  // also they are set to 0 if not provided by user
  static OutIndex getBiasesPassThroughIndex() { return 8; }

  // T7504
  bool isOutlineable() const override { return false; }

  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

  // getters for attributes
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

  ActivationFunction activation;
  ActivationFunction recurrent_activation;

  nonstd::optional<float> available_memory_proportion;
};

/**
 * Gradient operator for LSTM op
 */
class LSTMGradOp : public BaseOnnxRNNGradOp {
public:
  LSTMGradOp(const LSTMOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::map<int, int> &gradOutToNonGradIn() const final;

  bool hasLastCellStateGradInput() const;

  std::set<InIndex> optionalInputs() const final;

  // inputs 0-8 defined in BaseOnnxRNNGradOp
  static InIndex getInitialCInIndex() { return 9; }
  static InIndex getIntermediatesInIndex() { return 10; }
  static InIndex getLastCellStateGradInIndex() { return 11; }

  // outputs 0-4 are defined in BaseOnnxRNNGradOp
  static OutIndex getInitialCOutIndex() { return 5; }

  const bool hasInitialCInput;
  const std::string fwd_debug_name;
  const ActivationFunction activation;
  const ActivationFunction recurrent_activation;

private:
  // Populate inInfo with LSTM-specific mappings
  // Called in constructor
  void populateInInfo() override;

  // used to initialize fwdInitialCInInfo
  nonstd::optional<TensorInfo> getInitialCInInfo(const LSTMOp &fwd_op);

  // used to set inInfo for InitialC
  const nonstd::optional<TensorInfo> fwdInitialCInInfo;
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

/**
 * Gradient operator for PopartLSTMOp
 */
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
  // Populate inInfo to be used in initGradInputInfo
  // Called in constructor
  void populateInInfo();

  const TensorId forwardCellStateGradId;
  const ActivationFunction activation;
  const ActivationFunction recurrent_activation;

  // Return value for initGradInputInfo
  // Populated in constructor
  std::vector<GradInOutMapper> inInfoMapping;
};

} // namespace popart

#endif
