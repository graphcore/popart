#include <willow/error.hpp>
#include <willow/tensor.hpp>
#include <willow/varupdate.hpp>

namespace willow {
VarUpdateOp::VarUpdateOp(std::string op_type, TensorId varId_, Ir *pir)
    : Op({op_type, pir, {}, getWillowDomain()}), varId(varId_),
      varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  // void functions (like VarUpdateOp) have
  // no output tensors to set shapes for
}

void VarUpdateOp::imposeTopoCons() {
  input.tensor(getVarIndex())->consumers.setTopoLast(this);
}

int VarUpdateOp::getVarIndex() { return 0; }

int VarUpdateOp::getVarGradIndex() { return 1; }

SGDVarUpdateOp::SGDVarUpdateOp(TensorId varId_, Ir *pir)
    : VarUpdateOp("SGDVarUpdate", varId_, pir) {}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(*this));
}

int SGDVarUpdateOp::getLearnRateIndex() { return 2; }

ConstSGDVarUpdateOp::ConstSGDVarUpdateOp(TensorId varId_, Ir *pir, float lr_)
    : VarUpdateOp("ConstSGDVarUpdate", varId_, pir), learnRate(lr_) {}

float ConstSGDVarUpdateOp::getLearnRate() const { return learnRate; }

std::unique_ptr<Op> ConstSGDVarUpdateOp::clone() const {
  return std::unique_ptr<Op>(new ConstSGDVarUpdateOp(*this));
}

} // namespace willow
