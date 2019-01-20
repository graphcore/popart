#include <limits>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {
VarUpdateOp::VarUpdateOp(const OperatorIdentifier &_opid,
                         TensorId varId_,
                         Ir *_pir)
    : Op(_opid, _pir), varId(varId_), varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  // void functions (like VarUpdateOp) have
  // no output tensors to set shapes for
}

std::map<InIndex, Region>
VarUpdateOp::modifies(const std::map<InIndex, Shape> &) const {
  // Modifies the whole of the Var Tensor
  return {{getVarInIndex(), {true}}};
}

SGDVarUpdateOp::SGDVarUpdateOp(TensorId varId_, Ir *_pir)
    : VarUpdateOp(Onnx::CustomOperators::SgdVarUpdate, varId_, _pir) {}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return make_unique<SGDVarUpdateOp>(*this);
}

ConstSGDVarUpdateOp::ConstSGDVarUpdateOp(TensorId varId_, Ir *_pir, float lr_)
    : VarUpdateOp(Onnx::CustomOperators::ConstSgdVarUpdate, varId_, _pir),
      learnRate(lr_) {}

float ConstSGDVarUpdateOp::getLearnRate() const { return learnRate; }

std::unique_ptr<Op> ConstSGDVarUpdateOp::clone() const {
  return make_unique<ConstSGDVarUpdateOp>(*this);
}

namespace {
static GradOpCreator<SGDVarUpdateOp>
    sgdVarUpdateOpCreator(Onnx::CustomOperators::SgdVarUpdate);
static GradOpCreator<ConstSGDVarUpdateOp>
    constSgdVarUpdateOpCreator(Onnx::CustomOperators::ConstSgdVarUpdate);

} // namespace

} // namespace poponnx
