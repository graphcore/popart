#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

HostReduceGradCopyOp::HostReduceGradCopyOp(const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::HostReduceGradCopy, settings_) {}

void HostReduceGradCopyOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

std::unique_ptr<Op> HostReduceGradCopyOp::clone() const {
  return std::make_unique<HostReduceGradCopyOp>(*this);
}

HostSGD0VarUpdate::HostSGD0VarUpdate(const TensorId &varId_,
                                     OptimizerValue slr0,
                                     OptimizerValue wdsf0,
                                     const Op::Settings &settings_)
    : SGD0VarUpdateOpBase(Onnx::CustomOperators::HostSGD0VarUpdate,
                          varId_,
                          slr0,
                          wdsf0,
                          settings_) {}

std::unique_ptr<Op> HostSGD0VarUpdate::clone() const {
  return std::make_unique<HostSGD0VarUpdate>(*this);
}

namespace {} // namespace

} // namespace popart
