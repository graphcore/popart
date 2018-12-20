#ifndef GUARD_NEURALNET_RESHAPE_HPP
#define GUARD_NEURALNET_RESHAPE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// TODO: merge Reshape and Squeeze functionality (T5886)

// This Op is based on the ONNX Operator described at
// github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
// but it is slightly different: this Op is static w.r.t. shape
class ReshapeOp : public Op {
public:
  ReshapeOp(const OperatorIdentifier &_opid,
            Ir *_ir,
            const std::string &name = "",
            const Attributes &_attr = {});

  // This will be used by ReshapeGradOp
  ReshapeOp(const OperatorIdentifier &_opid, Ir *_ir, const Shape &outShape_);

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  virtual void connectInTensor(InIndex, TensorId) final;

  // The index at which the data is
  // input to this Op
  static InIndex getInIndex() { return 0; }

  // The output index of the unique, reshaped, output.
  static OutIndex getOutIndex() { return 0; }

  const Shape &getOutShape();

private:
  // The shape of the data output tensor
  Shape outShape;
};

// The gradient of reshape is the reverse of the
// reshape (which is a reshape)
class ReshapeGradOp : public ReshapeOp {
public:
  ReshapeGradOp(ReshapeOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
