#include <neuralnet/error.hpp>
#include <neuralnet/varupdate.hpp>

namespace neuralnet {
VarUpdateOp::VarUpdateOp(TensorId varId_, Graph *pgraph)
    : Op({"VarUpdate", pgraph, {}, getNeuralNetDomain()}), varId(varId_),
      varGradId(getGradId(varId)) {}

void VarUpdateOp::setup() {
  throw error("is there anything to do in var update op setup?");
}
} // namespace neuralnet
