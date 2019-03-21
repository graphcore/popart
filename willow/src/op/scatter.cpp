#include <algorithm>
#include <string>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/scatter.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ScatterOp::ScatterOp(const OperatorIdentifier &_opid,
                     int64_t axis_,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> ScatterOp::clone() const {
  return make_unique<ScatterOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScatterOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;

  result.push_back(make_unique<ScatterDataGradOp>(*this, axis));
  result.push_back(make_unique<ScatterUpdateGradOp>(*this, axis));

  return result;
}

int64_t ScatterOp::getAxis() const { return axis; }

void ScatterOp::setup() {
  int64_t axis_min = -static_cast<int64_t>(inShape(dataInIndex()).size());
  int64_t axis_max = inShape(dataInIndex()).size() - 1;
  if (axis_min > axis || axis > axis_max) {
    throw error(
        "GatherOp::setup axis = {} is outside the acceptable range [{}, {}]",
        axis,
        axis_min,
        axis_max);
  }

  if (inShape(indicesInIndex()) != inShape(updatesInIndex())) {
    throw error("ScatterOp::setup Mismatched indices and updates shape");
  }

  outInfo(outIndex()) = inInfo(dataInIndex());
}

void ScatterOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

ScatterDataGradOp::ScatterDataGradOp(const ScatterOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::ScatterDataGrad, op.getSettings()), axis(axis_) {}

std::unique_ptr<Op> ScatterDataGradOp::clone() const {
  return make_unique<ScatterDataGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScatterDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), ScatterOp::outIndex(), GradOpInType::GRADOUT},
      {indicesInIndex(), ScatterOp::indicesInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &ScatterDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), ScatterOp::dataInIndex()}};

  return outInfo;
}

void ScatterDataGradOp::setup() {
  outInfo(gradOutIndex()) = inInfo(gradInIndex());
}

int64_t ScatterDataGradOp::getAxis() const { return axis; }

void ScatterDataGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

ScatterUpdateGradOp::ScatterUpdateGradOp(const ScatterOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::ScatterUpdateGrad, op.getSettings()),
      axis(axis_) {}

std::unique_ptr<Op> ScatterUpdateGradOp::clone() const {
  return make_unique<ScatterUpdateGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScatterUpdateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), ScatterOp::outIndex(), GradOpInType::GRADOUT},
      {indicesInIndex(), ScatterOp::indicesInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &ScatterUpdateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), ScatterOp::updatesInIndex()}};

  return outInfo;
}

void ScatterUpdateGradOp::setup() {
  const auto type         = inInfo(gradInIndex()).dataType();
  outInfo(gradOutIndex()) = TensorInfo(type, inShape(indicesInIndex()));
}

int64_t ScatterUpdateGradOp::getAxis() const { return axis; }

void ScatterUpdateGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

namespace {
static OpCreator<ScatterOp> ScatterOpCreator(
    Onnx::Operators::Scatter_9,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis", 0);

      return std::unique_ptr<Op>(new ScatterOp(_opid, axis, settings));
    },
    true);
} // namespace

} // namespace poponnx
