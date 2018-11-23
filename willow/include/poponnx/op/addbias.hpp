#ifndef GUARD_NEURALNET_ADDBIAS_HPP
#define GUARD_NEURALNET_ADDBIAS_HPP

#include <poponnx/ir.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/op/reducesum.hpp>

namespace willow {

class ConvOp;

// A special purpose add operation used to add a bias to the output of a
// convolution operation.
class AddBiasOp : public Op {
public:
  AddBiasOp(ConvOp *);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places the data input at index 0, and the bias input
  // at index 1.
  static int dataInIndex();
  static int biasInIndex();
};

// The gradient op for the data input of the add bias op.
// Based on the identity op
class AddBiasDataGradOp : public IdentityOp {
public:
  AddBiasDataGradOp(AddBiasOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

// The gradient op for the bias input of the add bias op.
// Based on the reduce sum op.
class AddBiasBiasGradOp : public ReduceSumOp {
public:
  AddBiasBiasGradOp(AddBiasOp *, const std::vector<int64_t> &axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace willow

#endif
