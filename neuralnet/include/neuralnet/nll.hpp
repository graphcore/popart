#ifndef GUARD_NEURALNET_NLL_HPP
#define GUARD_NEURALNET_NLL_HPP

#include <neuralnet/graph.hpp>
#include <neuralnet/loss.hpp>

namespace neuralnet {

class NllLoss : public Loss {
public:
  virtual ~NllLoss() override = default;
  NllLoss(const std::string &argstring);
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  virtual std::unique_ptr<Op> getOp(Graph *) const override final;
  virtual std::string op_type() const override final;

  int probsIn() const;
  int labelsIn() const;
};

class NllOp : public Op {
public:
  NllOp(const OpConstructorBundle &, const NllLoss *nllloss);
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

} // namespace neuralnet

#endif
