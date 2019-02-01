#include <limits>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {
VarUpdateOp::VarUpdateOp(const OperatorIdentifier &_opid,
                         TensorId varId_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), varId(varId_), varGradId(getGradId(varId)) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  // void functions (like VarUpdateOp) have
  // no output tensors to set shapes for
}

// Modifies the whole of the Var Tensor
view::Region VarUpdateOp::modifies(InIndex index) const {
  if (index == getVarInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

SGDVarUpdateOp::SGDVarUpdateOp(TensorId varId_, const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::SgdVarUpdate, varId_, settings_) {}

std::unique_ptr<Op> SGDVarUpdateOp::clone() const {
  return make_unique<SGDVarUpdateOp>(*this);
}

ConstSGDVarUpdateOp::ConstSGDVarUpdateOp(TensorId varId_,
                                         float lr_,
                                         const Op::Settings &settings_)
    : VarUpdateOp(Onnx::CustomOperators::ConstSgdVarUpdate, varId_, settings_),
      learnRate(lr_) {}

float ConstSGDVarUpdateOp::getLearnRate() const { return learnRate; }

std::unique_ptr<Op> ConstSGDVarUpdateOp::clone() const {
  return make_unique<ConstSGDVarUpdateOp>(*this);
}

namespace {} // namespace

} // namespace poponnx
