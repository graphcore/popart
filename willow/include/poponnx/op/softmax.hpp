#ifndef GUARD_NEURALNET_SOFTMAX_HPP
#define GUARD_NEURALNET_SOFTMAX_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class NllLoss;

class SoftmaxOp : public ElementWiseUnaryOp {
public:
  SoftmaxOp(const OperatorIdentifier &_opid,
            Ir *_ir,
            const std::string &name = "",
            const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SoftmaxGradOp : public Op {
public:
  SoftmaxGradOp(SoftmaxOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  int gradProbsIn() const;
  int actsIn() const;

  static InIndex getGradProbsInIndex() { return 0; }
  static InIndex getActsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

class SoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  SoftmaxGradDirectOp(Ir *, const NllLoss *);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  const NllLoss *nlll() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  const NllLoss *nllloss_;
};

} // namespace poponnx

#endif
