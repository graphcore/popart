// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RNNBASE_HPP
#define GUARD_NEURALNET_RNNBASE_HPP

#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

// A base class for use by RNNOp, LSTMOp and GRUOp
class BaseOnnxRNNOp : public Op {
public:
  BaseOnnxRNNOp(const OperatorIdentifier &_opid,
                nonstd::optional<int64_t> hidden_size,
                const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override = 0;
  int64_t getMaxSeqLength() const;
  int64_t getBatchSize() const;
  int64_t getInputSize() const;

  int64_t getHiddenSize() const;
  // checks that the ONNX hidden_size attribute matches up with the input tensor
  // shapes
  void checkHiddenSize() const;

  // helpers to check if optional inputs are present
  bool hasBiasesInput() const;
  bool hasInitialHInput() const;
  bool hasSeqLenInput() const;

  std::set<InIndex> optionalInputs() const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // In the notation below, NUM_GATES is 1 for RNN, 3 for GRU and 4 for LSTM
  // X, [max_seq_length, batch_size, input_size]
  static InIndex getInputInIndex() { return 0; }
  // W, [num_directions, NUM_GATES*hidden_size, input_size], multiply with X_i
  static InIndex getInputWeightsInIndex() { return 1; }
  // R, [num_directions, NUM_GATES*hidden_size, hidden_size], multiply with H_i
  static InIndex getRecurrenceWeightsInIndex() { return 2; }
  // B, optional, [num_directions, 2*NUM_GATES*hidden_size], 0 if not specified
  static InIndex getBiasesInIndex() { return 3; }
  // sequence_lens, optional, [batch_size]
  static InIndex getSequenceLensInIndex() { return 4; }
  // initial_h, optional, [num_directions, batch_size, hidden_size]
  static InIndex getInitialHInIndex() { return 5; }

  // Y, optional, [max_seq_length, num_directions, batch_size, hidden_size]
  static OutIndex getFullHiddenStateOutIndex() { return 0; }
  // Y_h, optional, [num_directions, batch_size, hidden_size]. Y_h = Y[-1]
  static OutIndex getLastHiddenStateOutIndex() { return 1; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  virtual std::string getName() const { return "RNNBase"; }

  // getters for attributes
  nonstd::optional<int64_t> getHiddenSizeAttribute() const {
    return hidden_size_attribute;
  }

private:
  const nonstd::optional<int64_t> hidden_size_attribute;
};

// A base class for use by RNNGradOp, LSTMGradOp and and GRUGradOp
class BaseOnnxRNNGradOp : public Op {
public:
  BaseOnnxRNNGradOp(const OperatorIdentifier &_opid,
                    const BaseOnnxRNNOp &fwd_op);

  std::unique_ptr<Op> clone() const override = 0;
  void setup() override;
  const std::vector<GradInOutMapper> &gradInputInfo() const override;
  const std::map<int, int> &gradOutToNonGradIn() const override;

  bool hasLastHiddenStateGradInput() const;
  bool hasFullHiddenStateGradInput() const;

  static InIndex getInputInIndex() { return 0; }
  static InIndex getInputWeightsInIndex() { return 1; }
  static InIndex getRecurrenceWeightsInIndex() { return 2; }
  static InIndex getBiasesInIndex() { return 3; }
  static InIndex getInitialHInIndex() { return 4; }
  static InIndex getFullHiddenStateInIndex() { return 5; }
  static InIndex getLastHiddenStateGradInIndex() { return 6; }
  static InIndex getFullHiddenStateGradInIndex() { return 7; }
  static InIndex getSequenceLensInIndex() { return 8; }

  static OutIndex getInputOutIndex() { return 0; }
  static OutIndex getInputWeightsOutIndex() { return 1; }
  static OutIndex getRecurrenceWeightsOutIndex() { return 2; }
  static OutIndex getBiasesOutIndex() { return 3; }
  static OutIndex getInitialHOutIndex() { return 4; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const bool hasBiasesInput;
  const bool hasInitialHInput;
  const unsigned batch_size;
  const unsigned input_size;
  const unsigned max_seq_length;
  const unsigned hidden_size;
  const unsigned num_directions = 1;

protected:
  // conditionally get fwdBiasInInfo and fwdInitialHInInfo
  nonstd::optional<TensorInfo> getBiasInInfo(const BaseOnnxRNNOp &fwd_op);
  nonstd::optional<TensorInfo> getInitialHInInfo(const BaseOnnxRNNOp &fwd_op);

  // these are used in setup
  const TensorInfo fwdInputInInfo;
  const TensorInfo fwdInputWeightsInInfo;
  const TensorInfo fwdRecurrenceWeightsInInfo;
  // set to none if bias not provided by user
  const nonstd::optional<TensorInfo> fwdBiasInInfo;
  // set to none if initialH not provided by user
  const nonstd::optional<TensorInfo> fwdInitialHInInfo;
};

} // namespace popart

#endif
