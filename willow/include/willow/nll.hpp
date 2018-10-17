#ifndef GUARD_NEURALNET_NLL_HPP
#define GUARD_NEURALNET_NLL_HPP

#include <willow/ir.hpp>
#include <willow/loss.hpp>

namespace willow {

class NllLoss : public Loss {
public:
  NllLoss(TensorId probs, TensorId label, TensorId output);
  // label is the only streamed input tensor to this loss
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  virtual std::unique_ptr<Op> getOp(Ir *) const override final;
  virtual std::string op_type() const override final;
  int probsIn() const;
  int labelIn() const;
  TensorId probsTensorId() const;
  TensorId labelTensorId() const;

  virtual std::unique_ptr<Loss> clone() const override final {
    return std::unique_ptr<Loss>(new NllLoss(*this));
  }
};

class NllOp : public Op {
public:
  NllOp(const OpConstructorBundle &, const NllLoss *nllloss);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  const NllLoss *nlll() const;

private:
  const NllLoss *nllloss_;
};

class NllGradOp : public GradOp {
public:
  NllGradOp(NllOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;
  const NllLoss *nlll() const;

private:
  std::vector<GradInOutMapper> createNllLossGradInfo() const;
  std::map<int, int> createNllLossGradOutToIn() const;
  const NllLoss *nllloss_;
  OpId nllOpId;
};

} // namespace willow

#endif
