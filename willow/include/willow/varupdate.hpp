#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <willow/graph.hpp>

namespace willow {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(TensorId, Graph *);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual void setup() override final;
  static int getVarIndex();
  static int getVarGradIndex();
  static int getLearnRateIndex();
  void imposeTopoCons() override final;

private:
  TensorId varId;
  TensorId varGradId;
};
} // namespace willow

#endif
