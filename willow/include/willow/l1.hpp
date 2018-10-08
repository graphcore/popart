#ifndef GUARD_NEURALNET_L1_HPP
#define GUARD_NEURALNET_L1_HPP

#include <willow/graph.hpp>
#include <willow/loss.hpp>

namespace willow {

class L1Loss : public Loss {
public:
  L1Loss(const std::string &argstring);
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  virtual std::unique_ptr<Op> getOp(Graph *) const override final;
  virtual std::string op_type() const override final;

private:
  float lambda;
};

class L1Op : public Op {
public:
  L1Op(const OpConstructorBundle &, const L1Loss *l1loss);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  const L1Loss *l1l() const;

private:
  const L1Loss *l1loss_;
};

class L1GradOp : public GradOp {

public:
  L1GradOp(L1Op *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;
  const L1Loss *l1l() const;

private:
  std::vector<GradInOutMapper> createL1LossGradInfo() const;
  std::map<int, int> createL1LossGradOutToIn() const;
  OpId l1OpId;
  const L1Loss *l1loss_;
};

} // namespace willow

#endif
