// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRU_HPP
#define GUARD_NEURALNET_GRU_HPP

#include <popart/op.hpp>
#include <popart/op/rnnbase.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

/**
 * This op applies a single-layer GRU with a non-linearity to a batch of
 * input sequences. The op follows the ONNX specification described in
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU
 */
class GRUOp : public BaseOnnxRNNOp {
public:
  GRUOp(const OperatorIdentifier &_opid,
        nonstd::optional<int64_t> hidden_size,
        const std::string direction,
        bool linear_before_reset,
        const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  unsigned getNumChannels() const;

  int64_t getNumDirections() const;

  bool hasOutput(OutIndex) const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isTraining() const;

  // inputs 0-5 defined in BaseOnnxRNNOp

  // outputs 0-1 defined in BaseOnnxRNNOp
  // set to 0 if not provided by user
  static OutIndex getInitialHPassThroughIndex() { return 2; }
  // intermediate values needed to calculate the grad
  static OutIndex getIntermediatesPassThroughIndex() { return 3; }
  // restructured input weights to be used with poplar implementation
  static OutIndex getInputWeightsPassThroughIndex() { return 4; }
  // restructured recurrence weights to be used with poplar implementation
  static OutIndex getRecurrenceWeightsPassThroughIndex() { return 5; }
  // restructured biases to be used with poplar implementation
  // also they are set to 0 if not provided by user
  static OutIndex getBiasesPassThroughIndex() { return 6; }

  // TODO: T20922 : make this outlineable similar to LSTM
  bool isOutlineable() const override { return false; }

  // getters for attributes
  std::string getDirectionAttribute() const { return direction_attribute; }
  int getLinearBeforeResetAttribute() const {
    return linear_before_reset_attribute;
  }

private:
  void maybeCreatePassThroughOutput(const TensorId &new_id,
                                    OutIndex pass_through_index,
                                    const TensorInfo &out_info);
  int getNumIntermediates() { return 3 + (linear_before_reset_attribute != 0); }
  int getNumBiases() { return 3 * (1 + (linear_before_reset_attribute != 0)); }
  void trySetOutInfo(OutIndex, const TensorInfo &);

  const std::string direction_attribute   = "forward";
  const int linear_before_reset_attribute = 0;
};

/**
 * Gradient operator for GRUOp
 */
class GRUGradOp : public BaseOnnxRNNGradOp {
public:
  GRUGradOp(const GRUOp &);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;

  std::set<InIndex> optionalInputs() const final;

  // Inputs 0-8 are defined in BaseOnnxRNNGradOp
  static InIndex getIntermediatesInIndex() { return 9; }

  // Outputs 0-4 are defined in BaseOnnxRNNGradOp

  const unsigned linear_before_reset_attribute;
};

} // namespace popart

#endif
