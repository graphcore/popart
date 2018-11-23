#ifndef GUARD_NEURALNET_NLL_HPP
#define GUARD_NEURALNET_NLL_HPP

#include <poponnx/ir.hpp>
#include <poponnx/op/loss.hpp>

namespace willow {

class NllLoss : public Loss {
public:
  NllLoss(TensorId probs, TensorId label, TensorId output);
  // label is the only streamed input tensor to this loss
  std::vector<TensorId> getStreamTensorNames() const final;
  std::unique_ptr<Op> getOp(Ir *) const final;
  std::string op_type() const final;
  int probsIn() const;
  int labelIn() const;
  TensorId probsTensorId() const;
  TensorId labelTensorId() const;
  std::unique_ptr<Loss> clone() const final;
};

class NllOp : public LossOp {
public:
  NllOp(const OpConstructorBundle &, const NllLoss *nllloss);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  const NllLoss *nlll() const;

private:
  const NllLoss *nllloss_;
};

class NllGradOp : public Op {
public:
  NllGradOp(NllOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  const NllLoss *nlll() const;

private:
  const NllLoss *nllloss_;
};

} // namespace willow

#endif
