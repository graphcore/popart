// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/copyvarupdate.hpp>
#include <popart/opmanager.hpp>

namespace popart {

CopyVarUpdateOp::CopyVarUpdateOp(TensorId varId_, const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::CopyVarUpdate,
                             varId_,
                             settings_) {}

std::unique_ptr<Op> CopyVarUpdateOp::clone() const {
  return std::make_unique<CopyVarUpdateOp>(*this);
}

view::Regions CopyVarUpdateOp::modifies(InIndex index) const {
  if (index == getVarToUpdateInIndex()) {
    // Modifies differs from base class since copy will
    // overwrite the tensor to update completely
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(index))};
  }
}

namespace {} // namespace

} // namespace popart
