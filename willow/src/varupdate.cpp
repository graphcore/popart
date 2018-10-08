#include <willow/error.hpp>
#include <willow/tensor.hpp>
#include <willow/varupdate.hpp>

namespace willow {
VarUpdateOp::VarUpdateOp(TensorId varId_, Graph *pgraph)
    : Op({"VarUpdate", pgraph, {}, getWillowDomain()}), varId(varId_),
      varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

std::unique_ptr<Op> VarUpdateOp::clone() const {
  return std::unique_ptr<Op>(new VarUpdateOp(*this));
}

void VarUpdateOp::setup() {
  // throw error("is there anything to do in var update op setup?");
}

int VarUpdateOp::getVarIndex() { return 0; }

int VarUpdateOp::getVarGradIndex() { return 1; }

int VarUpdateOp::getLearnRateIndex() { return 2; }

void VarUpdateOp::imposeTopoCons() {
  input.tensor(getVarIndex())->consumers.setTopoLast(this);
}

} // namespace willow
