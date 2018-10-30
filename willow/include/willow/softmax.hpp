#ifndef GUARD_NEURALNET_SOFTMAX_HPP
#define GUARD_NEURALNET_SOFTMAX_HPP

#include <willow/ir.hpp>

namespace willow {

class NllLoss;

class SoftmaxOp : public Op {
public:
  SoftmaxOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

class SoftmaxGradOp : public GradOp {
public:
  SoftmaxGradOp(SoftmaxOp *);
  virtual Op *getNonGradCreator() const override final;
  // equivalent of getNonGradCreator, but no downcasting
  SoftmaxOp *getSoftmaxOp() const;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;
  int gradProbsIn() const;
  int actsIn() const;

private:
  std::vector<GradInOutMapper> createSoftmaxGradInfo() const;
  std::map<int, int> createSoftmaxGradOutToIn() const;
  SoftmaxOp *softmaxOp;
};

// not a gradient of a single Op, so not inheriting from GradOp
class SoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  SoftmaxGradDirectOp(Op *, const NllLoss *);
  virtual std::unique_ptr<Op> clone() const override final;
  // this Op has no Grad Ops, throw error if called
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  SoftmaxOp *getSoftmaxOp() const;
  const NllLoss *nlll() const;
  //  int labelIn() const;
  //  TensorId labelInId() const;
  //  int probsIn() const;
  //  TensorId probsInId() const;

private:
  SoftmaxOp *softmaxOp;
  const NllLoss *nllloss_;
};

} // namespace willow

#endif
