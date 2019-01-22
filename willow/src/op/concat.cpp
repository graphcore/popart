#include <algorithm>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

std::unique_ptr<RegionIOMap>
ConcatInplaceOp::aliases(const std::map<InIndex, Shape> &shapesToConcat) const {

  // we should do a test here that the
  // shapes above can be concatenated
  // and a test that the indices are contiguous from 0

  std::map<InIndex, RegionIO> aliases;
  for (auto index_shape : shapesToConcat) {
    auto index = index_shape.first;

    // the whole region of the input will be aliased,
    Region inRegion{true};

    // and for now we'll say that the whole of the output is aliased to
    // the input slice, although we can improve this in the future
    Region outRegion{true};

    aliases.emplace(std::pair<InIndex, RegionIO>(index, {inRegion, outRegion}));
  }
  return std::unique_ptr<RegionIOMap>(new RegionIOMap(std::move(aliases)));
}

ConcatOp::ConcatOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

ConcatInplaceOp::ConcatInplaceOp(const ConcatOp &op, int64_t axis_)
    : ConcatOp(Onnx::CustomOperators::ConcatInplace, axis_, op.getSettings()) {}

std::unique_ptr<Op> ConcatOp::clone() const {
  return make_unique<ConcatOp>(*this);
}

std::unique_ptr<Op>
ConcatOp::getInplaceVariant(const OperatorIdentifier &operator_id,
                            const std::vector<InIndex> &inIndices) {

  auto validVariants = inplaceVariants(inIndices);

  if (std::find(validVariants.begin(),
                validVariants.end(),
                Onnx::CustomOperators::ConcatInplace) != validVariants.end()) {
    return make_unique<ConcatInplaceOp>(*this, axis);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id, inIndices);
}

std::unique_ptr<Op> ConcatInplaceOp::clone() const {
  return make_unique<ConcatInplaceOp>(*this);
}

int64_t ConcatOp::getAxis() const { return axis; }

std::vector<OperatorIdentifier>
ConcatOp::inplaceVariants(const std::vector<InIndex> &inIndices) const {
  if (inIndices.size() != input->n()) {
    return {};
  }

  for (auto index : inIndices) {
    if (!input->hasIndex(index)) {
      return {};
    }
  }

  return {Onnx::CustomOperators::ConcatInplace};
}

void ConcatOp::setup() {
  const auto input_count = input->n();

  if (input_count == 0) {
    throw error("Cannot concat zero tensors");
  }

  const DataType outType = inInfo(getInIndex(0)).dataType();
  Shape outShape         = inShape(getInIndex(0));
  outShape[axis]         = 0;

  for (int i = 0; i < input_count; ++i) {
    const auto shape = inShape(getInIndex(i));

    if (outShape.size() != shape.size()) {
      throw error("Input {} to concat {}({}, {}) doesn't have matching rank",
                  i,
                  id,
                  opid,
                  name());
    }

    auto outShapePrefixBegin = std::begin(outShape);
    auto outShapePrefixEnd   = std::begin(outShape) + axis;
    auto outShapeSuffixBegin = std::begin(outShape) + axis + 1;
    auto outShapeSuffixEnd   = std::end(outShape);

    auto shapePrefixBegin = std::begin(shape);
    auto shapeSuffixBegin = std::begin(shape) + axis + 1;

    if (!std::equal(outShapePrefixBegin, outShapePrefixEnd, shapePrefixBegin) ||
        !std::equal(outShapeSuffixBegin, outShapeSuffixEnd, shapeSuffixBegin)) {
      throw error("Input {} to concat {}({}, {}) doesn't have matching shape",
                  i,
                  id,
                  opid,
                  name());
    }

    outShape[axis] += shape[axis];
  }

  outInfo(getOutIndex()) = TensorInfo(outType, outShape);
}

std::vector<std::unique_ptr<Op>> ConcatOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.reserve(input->n());

  for (int i = 0; i < input->n(); ++i) {
    result.push_back(make_unique<ConcatGradOp>(*this, i));
  }

  return result;
}

ConcatGradOp::ConcatGradOp(const ConcatOp &fwd, InIndex inputIndex)
    : Op(Onnx::GradOperators::ConcatGrad, fwd.getSettings()),
      axis(fwd.getAxis()), start(0), end(0), fwdInput(inputIndex) {
  for (int i = 0; i < inputIndex; ++i) {
    auto shape = fwd.inShape(ConcatOp::getInIndex(i));
    start += shape[axis];
  }

  for (int i = 0; i <= inputIndex; ++i) {
    auto shape = fwd.inShape(ConcatOp::getInIndex(i));
    end += shape[axis];
  }

  const DataType outType =
      fwd.inInfo(ConcatOp::getInIndex(fwdInput)).dataType();
  Shape outShape = fwd.inShape(ConcatOp::getInIndex(fwdInput));
  outShape[axis] = end - start;

  gradInfo               = TensorInfo(outType, outShape);
  gradOutToNonGradInInfo = {{getOutIndex(), ConcatOp::getInIndex(fwdInput)}};
}

void ConcatOp::appendAttributes(std::stringstream &ss,
                                const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "axis", axis);
}

ConcatGradOp::ConcatGradOp(const OperatorIdentifier &_opid,
                           const ConcatGradOp &concat_grad_op)
    : Op(_opid, concat_grad_op.getSettings()), axis(concat_grad_op.axis),
      start(concat_grad_op.start), end(concat_grad_op.end),
      fwdInput(concat_grad_op.fwdInput), gradInfo(concat_grad_op.gradInfo),
      gradOutToNonGradInInfo(concat_grad_op.gradOutToNonGradInInfo) {}

std::unique_ptr<Op> ConcatGradOp::clone() const {
  return make_unique<ConcatGradOp>(*this);
}

const std::vector<GradInOutMapper> &ConcatGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ConcatOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &ConcatGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

void ConcatGradOp::setup() { outInfo(getOutIndex()) = gradInfo; }

int64_t ConcatGradOp::getAxis() const { return axis; }

int64_t ConcatGradOp::getStart() const { return start; }

int64_t ConcatGradOp::getEnd() const { return end; }

namespace {

static OpCreator<ConcatOp> concatOpCreator(
    Onnx::Operators::Concat_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis");

      return std::unique_ptr<Op>(new ConcatOp(_opid, axis, settings));
    },
    true);

} // namespace

} // namespace poponnx
