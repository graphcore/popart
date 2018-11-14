#ifndef GUARD_NEURALNET_L1_HPP
#define GUARD_NEURALNET_L1_HPP

#include <poponnx/ir.hpp>
#include <poponnx/loss.hpp>

namespace willow {

class L1Loss : public Loss {
public:
  // where lambda*|"input"|_1 = "output" (so output has rank 0)
  L1Loss(TensorId input, TensorId output, float lambda);
  // There are no tensors streamed into this loss layer (unlike NLL for
  // example which has a label streamed in)
  virtual std::vector<TensorId> getStreamTensorNames() const override final;
  virtual std::unique_ptr<Op> getOp(Ir *) const override final;
  virtual std::string op_type() const override final;
  TensorId getInputId() const;
  float getLambda() const;
  virtual std::unique_ptr<Loss> clone() const override final {
    return std::unique_ptr<Loss>(new L1Loss(*this));
  }

private:
  float lambda;
};

class L1Op : public LossOp {
public:
  L1Op(const OpConstructorBundle &, const L1Loss *l1loss);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  const L1Loss *l1l() const;

private:
  const L1Loss *l1loss_;
};

class L1GradOp : public Op {

public:
  L1GradOp(L1Op *);
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;
  const L1Loss *l1l() const;

private:
  const L1Loss *l1loss_;
};

} // namespace willow

#endif
