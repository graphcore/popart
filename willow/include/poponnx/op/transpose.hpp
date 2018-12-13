#ifndef GUARD_NEURALNET_TRANSPOSE_HPP
#define GUARD_NEURALNET_TRANSPOSE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Corresponds to the ONNX Transpose op
// for N-dimensional tensors.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose
class TransposeOp : public Op {
public:
  TransposeOp(const OpConstructorBundle &, const std::vector<int64_t> &perm);
  TransposeOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() final;

  const std::vector<int64_t> &getPerm() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // Get the permutation required to reverse the Transpose operation
  std::vector<int64_t> generateReversePermutation() const;

private:
  // the new permutation of the tensor axes
  std::vector<int64_t> perm;
  void setDefaultPerm();
};

// TransposeGrad is a reverse transposition
class TransposeGradOp : public TransposeOp {
public:
  TransposeGradOp(TransposeOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
