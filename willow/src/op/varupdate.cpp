#include <poponnx/error.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/tensor.hpp>

namespace willow {
VarUpdateOp::VarUpdateOp(std::string op_type, TensorId varId_, Ir *_pir)
    : Op({op_type, _pir, {}, getPoponnxDomain()}), varId(varId_),
      varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  // void functions (like VarUpdateOp) have
  // no output tensors to set shapes for
}

int VarUpdateOp::getVarIndex() { return 0; }

int VarUpdateOp::getVarGradIndex() { return 1; }

SGDVarUpdateOp::SGDVarUpdateOp(TensorId varId_, Ir *_pir)
    : VarUpdateOp("SGDVarUpdate", varId_, _pir) {}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return std::unique_ptr<Op>(new SGDVarUpdateOp(*this));
}

int SGDVarUpdateOp::getLearnRateIndex() { return 2; }

ConstSGDVarUpdateOp::ConstSGDVarUpdateOp(TensorId varId_, Ir *_pir, float lr_)
    : VarUpdateOp("ConstSGDVarUpdate", varId_, _pir), learnRate(lr_) {}

float ConstSGDVarUpdateOp::getLearnRate() const { return learnRate; }

std::unique_ptr<Op> ConstSGDVarUpdateOp::clone() const {
  return std::unique_ptr<Op>(new ConstSGDVarUpdateOp(*this));
}

} // namespace willow
