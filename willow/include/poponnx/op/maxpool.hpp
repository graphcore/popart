#ifndef GUARD_NEURALNET_MAXPOOL_HPP
#define GUARD_NEURALNET_MAXPOOL_HPP

#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/receptive.hpp>

namespace willow {

// c++ note : the conditions are suitable here
// for the compiler to generate defaults for
// "the 3": destructor, copy constructor, assigment op.
class MaxPoolOp : public HasReceptiveFieldOp {
public:
  MaxPoolOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

private:
  void setup0() final;
  void setSpatialK() final;
};

class MaxPoolGradOp : public Op {
public:
  MaxPoolGradOp(MaxPoolOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  int getPrePooledIn() const;
  int getPooledIn() const;
  int getGradPooledIn() const;
  const MaxPoolOp *getCloneOfCreator();

private:
  // The shape and type of the input to the
  // forward op which creates this backwards op
  TensorInfo unpooledInfo;
  // A copy of the forward op which creates
  // this backwards op. Note
  // 1) backends will need a copy of this op to determine
  //    how to do the backwards pass (padding, striding, etc)
  // 2) we DON'T store a pointer to the creating forward op,
  //    which might be optimised out and deleted
  std::unique_ptr<Op> cloneOfCreator;
};

} // namespace willow

#endif
