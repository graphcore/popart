#include <algorithm>
#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/instancenorm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

InstanceNormOp::InstanceNormOp(const OperatorIdentifier &_opid,
                               float _epsilon,
                               const Op::Settings &settings_)
    : Op(_opid, settings_), epsilon(_epsilon) {}

std::unique_ptr<Op> InstanceNormOp::clone() const {
  return make_unique<InstanceNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> InstanceNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  throw error("GradOps not support for InstanceNorm");
}

void InstanceNormOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInputInIndex());
}

void InstanceNormOp::appendAttributes(std::stringstream &ss,
                                      const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "epsilon", epsilon);
}

namespace {
static OpCreator<InstanceNormOp> instanceNormOpCreator(
    Onnx::Operators::InstanceNormalization_6,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      // default epsilon is 10**(-5)
      float epsilon = attr.getAttribute<Attributes::Float>("epsilon", 1e-5f);

      return std::unique_ptr<Op>(new InstanceNormOp(_opid, epsilon, settings));
    },
    true);

} // namespace

} // namespace poponnx
