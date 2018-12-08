#include <limits>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {
VarUpdateOp::VarUpdateOp(OpType op_type, TensorId varId_, Ir *_pir)
    : Op({op_type, _pir, {}}), varId(varId_), varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  // void functions (like VarUpdateOp) have
  // no output tensors to set shapes for
}

bool VarUpdateOp::modifies(InIndex in_index) const {
  return in_index == getVarInIndex();
}

SGDVarUpdateOp::SGDVarUpdateOp(TensorId varId_, Ir *_pir)
    : VarUpdateOp(OpType::SGDVARUPDATE, varId_, _pir) {}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return make_unique<SGDVarUpdateOp>(*this);
}

ConstSGDVarUpdateOp::ConstSGDVarUpdateOp(TensorId varId_, Ir *_pir, float lr_)
    : VarUpdateOp(OpType::CONSTSGDVARUPDATE, varId_, _pir), learnRate(lr_) {}

float ConstSGDVarUpdateOp::getLearnRate() const { return learnRate; }

std::unique_ptr<Op> ConstSGDVarUpdateOp::clone() const {
  return make_unique<ConstSGDVarUpdateOp>(*this);
}

} // namespace poponnx
