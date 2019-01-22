#include <algorithm>
#include <string>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GatherOp::GatherOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> GatherOp::clone() const {
  return make_unique<GatherOp>(*this);
}

std::vector<std::unique_ptr<Op>> GatherOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(make_unique<GatherGradOp>(*this, axis));
  return result;
}

int64_t GatherOp::getAxis() const { return axis; }

void GatherOp::setup() {
  int64_t axis_min = -static_cast<int64_t>(inShape(dataInIndex()).size());
  int64_t axis_max = inShape(dataInIndex()).size() - 1;
  if (axis_min > axis || axis > axis_max) {
    throw error(
        "GatherOp::setup axis = {} is outside the acceptable range [{}, {}]",
        axis,
        axis_min,
        axis_max);
  }

  // ONNX allows the axis attribute to be negative
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [-m+1, m-1]
  axis += inShape(dataInIndex()).size();       // axis in the range [0, 2m-1]
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [0, m-1]

  // Replace the axis dimension with the indices shape
  auto data_shape            = inShape(dataInIndex());
  const auto indices_shape   = inShape(indicesInIndex());
  const auto insertion_point = data_shape.erase(data_shape.begin() + axis);

  data_shape.insert(
      insertion_point, indices_shape.begin(), indices_shape.end());

  // Use the computed shape with the data input type
  outInfo(outIndex()) =
      TensorInfo(inInfo(dataInIndex()).dataType(), data_shape);
}

void GatherOp::appendAttributes(std::stringstream &ss,
                                const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "axis", axis);
}

GatherGradOp::GatherGradOp(const GatherOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::GatherGrad, op.getSettings()), axis(axis_),
      fwdDataInfo(op.inInfo(GatherOp::dataInIndex())) {}

std::unique_ptr<Op> GatherGradOp::clone() const {
  return make_unique<GatherGradOp>(*this);
}

const std::vector<GradInOutMapper> &GatherGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), GatherOp::outIndex(), GradOpInType::GRADOUT},
      {indicesInIndex(), GatherOp::indicesInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &GatherGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), GatherOp::dataInIndex()}};

  return outInfo;
}

void GatherGradOp::setup() { outInfo(gradOutIndex()) = fwdDataInfo; }

int64_t GatherGradOp::getAxis() const { return axis; }

namespace {
static OpCreator<GatherOp> gatherOpCreator(
    Onnx::Operators::Gather_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis");

      return std::unique_ptr<Op>(new GatherOp(_opid, axis, settings));
    },
    true);
} // namespace

} // namespace poponnx
