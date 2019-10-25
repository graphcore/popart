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

namespace {} // namespace

} // namespace popart
