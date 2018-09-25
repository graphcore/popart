#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(TensorId, Graph *);
  virtual void setup() override final;

private:
  TensorId varId;
  TensorId varGradId;
};
} // namespace neuralnet

#endif
