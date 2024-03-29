// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RNN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RNN_HPP_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/lstmutil.hpp>
#include <popart/op/rnnbase.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

/**
 * This op applies a single-layer Elman RNN with a non-linearity to a batch of
 * input sequences. The op follows the ONNX specification described in
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
 *
 * For each batch element, the following output is computed:
 * \f[
 *   h_t = f(W x_t + b_x + R h_{t-1} + b_h)
 * \f]
 * where:
 * - \f$f\f$ is a supported nonlinearity function
 * - \f$W\f$ is the input weight
 * - \f$x_t\f$ is the t'th element of the input sequence
 * - \f$R\f$ is the recurrence weight matrix
 * - \f$h_{t-1}\f$ is the previous output sequence element. \f$h_0\f$ can be
 * provided by the user
 * - \f$b_x\f$ and \f$b_h\f$ are the input and recurrence biases respectively
 *
 * The op outputs the full sequence \f$h_1, h_2, ...\f$, as well as the last
 * element of the sequence.
 *
 * If the biases or \f$h_0\f$ are not set, they are considered to be 0 and not
 * trained (are treated as constant 0s in the model).
 */
class RNNOp : public BaseOnnxRNNOp {
public:
  RNNOp(const OperatorIdentifier &_opid,
        ActivationFunction activation,
        nonstd::optional<int64_t> hidden_size,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  int getInBatchAxis(InIndex) const override;
  int getOutBatchAxis(OutIndex) const override;

  // inputs 0-5 defined in BaseOnnxRNNOp
  // outputs 0-1 defined in BaseOnnxRNNOp

  bool isOutlineable() const override { return true; }

  virtual std::string getName() const final { return "RNN"; }

  const ActivationFunction activation_attribute;
};

/**
 * Gradient operator for RNNOp
 */
class RNNGradOp : public BaseOnnxRNNGradOp {
public:
  RNNGradOp(const RNNOp &);
  std::unique_ptr<Op> clone() const final;

  std::set<InIndex> optionalInputs() const final;

  // inputs 0-8 defined in BaseOnnxRNNGradOp
  // outputs 0-4 defined in BaseOnnxRNNGradOp

  const ActivationFunction activation_attribute;

private:
  // Populate inInfo with RNN-specific mappings
  // Called in constructor
  void populateInInfo() override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RNN_HPP_
