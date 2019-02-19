#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

FlattenAliasOp::FlattenAliasOp(const OperatorIdentifier &_opid,
                               int64_t axis_,
                               const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> FlattenAliasOp::clone() const {
  return make_unique<FlattenAliasOp>(*this);
}

std::unique_ptr<Op> FlattenOp::clone() const {
  return make_unique<FlattenOp>(*this);
}

void FlattenAliasOp::setup() {
  const auto in_shape = inInfo(getInIndex()).shape();
  const auto begin    = in_shape.begin();
  const auto mid      = in_shape.begin() + axis;
  const auto end      = in_shape.end();

  // The product of the first axis dimensions to flatten
  const auto m = std::accumulate(begin, mid, 1, std::multiplies<int64_t>());

  // The product of the remaining dimensions
  const auto n = std::accumulate(mid, end, 1, std::multiplies<int64_t>());

  // The "flattened" shape
  const Shape out_shape = {m, n};

  outInfo(getOutIndex()) = {inInfo(getInIndex()).data_type(), out_shape};
}

std::vector<std::unique_ptr<Op>> FlattenAliasOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;

  result.push_back(make_unique<FlattenGradOp>(*this));

  return result;
}

int64_t FlattenAliasOp::getAxis() const { return axis; }

void FlattenAliasOp::setAxis(int64_t value) { axis = value; }

void FlattenAliasOp::appendAttributes(std::stringstream &ss,
                                      const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "axis", axis);
}

FlattenGradOp::FlattenGradOp(const FlattenAliasOp &fwdOp)
    : ReshapeOp(Onnx::GradOperators::FlattenGrad,
                fwdOp.inShape(FlattenAliasOp::getInIndex()),
                fwdOp.getSettings()) {}

const std::vector<GradInOutMapper> &FlattenGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), FlattenAliasOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &FlattenGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), FlattenAliasOp::getInIndex()}};
  return outInfo;
}

namespace {
static std::unique_ptr<Op> flattenOpFactory(const OperatorIdentifier &_opid,
                                            const Op::Settings &settings,
                                            const Attributes &attr) {
  int64_t axis = attr.getAttribute<Attributes::Int>("axis", 1);

  return make_unique<FlattenOp>(_opid, axis, settings);
}

static OpCreator<FlattenOp> flattenOpCreator({Onnx::Operators::Flatten_1,
                                              Onnx::Operators::Flatten_9},
                                             flattenOpFactory,
                                             true);
} // namespace

} // namespace poponnx
