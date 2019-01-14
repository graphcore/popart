#include <algorithm>
#include <string>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/scatter.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ScatterOp::ScatterOp(const OperatorIdentifier &_opid,
                     Ir *_ir,
                     const std::string &name,
                     const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  nAtts.setIfPresent(axis, "axis");
}

std::unique_ptr<Op> ScatterOp::clone() const {
  return make_unique<ScatterOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScatterOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;

  result.push_back(make_unique<ScatterDataGradOp>(this, axis));
  result.push_back(make_unique<ScatterUpdateGradOp>(this, axis));

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

ScatterDataGradOp::ScatterDataGradOp(ScatterOp *op, int64_t axis_)
    : Op({Onnx::GradOperators::ScatterDataGrad, op->pir, {}}), axis(axis_) {}

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

ScatterUpdateGradOp::ScatterUpdateGradOp(ScatterOp *op, int64_t axis_)
    : Op({Onnx::GradOperators::ScatterUpdateGrad, op->pir, {}}), axis(axis_) {}

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

namespace {
static OpCreator<ScatterOp> ScatterOpCreator(Onnx::Operators::Scatter);
static GradOpCreator<ScatterDataGradOp>
    scatterDataGradOpCreator(Onnx::GradOperators::ScatterDataGrad);
static GradOpCreator<ScatterUpdateGradOp>
    scatterUpdateGradOpCreator(Onnx::GradOperators::ScatterUpdateGrad);
} // namespace

} // namespace poponnx
