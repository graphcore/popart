#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(TensorId, Graph *);
  virtual void setup() override final;
  static int getVarIndex();
  static int getVarGradIndex();
  static int getLearnRateIndex();
  void imposeTopoCons() override final;
  // very high priority, so that performed as early as possible
  virtual double priority() const override final {
    return std::numeric_limits<double>::max();
  }

private:
  TensorId varId;
  TensorId varGradId;
};
} // namespace neuralnet

#endif
