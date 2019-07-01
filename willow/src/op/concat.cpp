#include <algorithm>

#include <memory>
#include <poponnx/op/concat.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

ConcatInplaceOp::ConcatInplaceOp(int64_t axis_, const Op::Settings &settings)
    : ConcatOp(Onnx::CustomOperators::ConcatInplace, axis_, settings) {}

ConcatOp::ConcatOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

ConcatInplaceOp::ConcatInplaceOp(const ConcatOp &op, int64_t axis_)
    : ConcatOp(Onnx::CustomOperators::ConcatInplace, axis_, op.getSettings()) {}

std::unique_ptr<Op> ConcatOp::clone() const {
  return std::make_unique<ConcatOp>(*this);
}

void ConcatOp::regMapPreChecks(InIndex inIndex) const {
  if (outOffsets.size() != input->tensorMap().size() + 1) {
    throw error(
        "ILE in ConcatOp::(fwd/bwd)RegMap, outOffsets not set correctly. It "
        "has size {} and the input tensorMap has size {}, this for Op {}",
        outOffsets.size(),
        input->tensorMap().size(),
        str());
  }
  if (inIndex >= input->tensorMap().size() || inIndex < 0) {
    throw error("invalid index in ConcatOp::fwdRegMap");
  }
}

view::RegMap ConcatOp::fwdRegMap(InIndex inIndex) const {
  regMapPreChecks(inIndex);
  int64_t offset = outOffsets[inIndex];
  int64_t axisIn = getAxis();
  return [axisIn, offset](const view::Region &r_in) {
    view::LowBounds lower = r_in.getLower();
    view::UppBounds upper = r_in.getUpper();
    lower[axisIn] += offset;
    upper[axisIn] += offset;
    return view::Region(lower, upper);
  };
}

view::RegMap ConcatOp::bwdRegMap(InIndex inIndex) const {
  regMapPreChecks(inIndex);
  int64_t offset = outOffsets[inIndex];
  int64_t axisIn = getAxis();
  return [axisIn, offset](const view::Region &r_out) {
    view::LowBounds lower = r_out.getLower();
    view::UppBounds upper = r_out.getUpper();
    lower[axisIn] -= offset;
    upper[axisIn] -= offset;
    // TODO T8446 : check intersect?
    return view::Region(lower, upper);
  };
}

std::unique_ptr<Op>
ConcatOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ConcatInplace) {
    return std::make_unique<ConcatInplaceOp>(*this, axis);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::unique_ptr<Op> ConcatInplaceOp::clone() const {
  return std::make_unique<ConcatInplaceOp>(*this);
}

int64_t ConcatOp::getAxis() const { return axis; }

std::vector<std::tuple<OperatorIdentifier, float>>
ConcatOp::inplacePriorityDefault() const {
  // see T6768: choosing default priorities
  return {{Onnx::CustomOperators::ConcatInplace, 10.0f}};
}

Shape ConcatOp::getOutputShape(int64_t axis,
                               const std::vector<const Shape *> inputs) {
  Shape outShape(*inputs[0]);
  outShape[axis] = 0;

  for (int i = 0; i < inputs.size(); i++) {
    const auto &shape = *inputs[i];

    if (outShape.size() != shape.size()) {
      throw error(
          "Input {} of {} to concat does not have matching output rank. "
          "Input {} has rank {} while the output has rank {}",
          i,
          inputs.size(),
          i,
          outShape.size(),
          shape.size());
    }

    auto outShapePrefixBegin = std::begin(outShape);
    auto outShapePrefixEnd   = std::begin(outShape) + axis;
    auto outShapeSuffixBegin = std::begin(outShape) + axis + 1;
    auto outShapeSuffixEnd   = std::end(outShape);

    auto shapePrefixBegin = std::begin(shape);
    auto shapeSuffixBegin = std::begin(shape) + axis + 1;

    if (!std::equal(outShapePrefixBegin, outShapePrefixEnd, shapePrefixBegin) ||
        !std::equal(outShapeSuffixBegin, outShapeSuffixEnd, shapeSuffixBegin)) {
      std::stringstream ss;
      ss << "In ConcatOp::getOutputShape, axis = " << axis << " and shapes : ";
      for (auto sp : inputs) {
        ss << '\n';
        appendSequence(ss, *sp);
      }

      throw error(
          "Input {} to concat does not have matching shape. {}", i, ss.str());
    }

    outShape[axis] += shape[axis];
  }

  return outShape;
}

void ConcatOp::setup() {
  const auto input_count = input->n();

  if (input_count == 0) {
    throw error("Cannot concat zero tensors");
  }

  const DataType outType = inInfo(getInIndex(0)).dataType();
  std::vector<const Shape *> inputs;
  inputs.reserve(input_count);
  for (int i = 0; i < input_count; i++) {
    inputs.push_back(&inShape(getInIndex(i)));
  }
  Shape outShape;
  try {
    outShape = getOutputShape(axis, inputs);
  } catch (const error &) {
    logging::op::err(
        "Error trying to calculate output shape for concat {}({}, {})",
        id,
        opid,
        name());
    throw;
  }

  outOffsets = {0};
  for (int i = 0; i < input_count; ++i) {
    const auto shape = inShape(getInIndex(i));
    outOffsets.push_back(outOffsets.back() + shape[axis]);
  }

  outInfo(getOutIndex()) = TensorInfo(outType, outShape);
}

std::vector<std::unique_ptr<Op>> ConcatOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.reserve(input->n());

  for (int i = 0; i < input->n(); ++i) {
    result.push_back(std::make_unique<ConcatGradOp>(*this, i));
  }

  return result;
}

// Concat can be replace by identity if there is 1 input
bool ConcatOp::canBeReplacedByIdentity() { return input->n() == 1; }

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

void ConcatOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

ConcatGradOp::ConcatGradOp(const OperatorIdentifier &_opid,
                           const ConcatGradOp &concat_grad_op)
    : Op(_opid, concat_grad_op.getSettings()), axis(concat_grad_op.axis),
      start(concat_grad_op.start), end(concat_grad_op.end),
      fwdInput(concat_grad_op.fwdInput), gradInfo(concat_grad_op.gradInfo),
      gradOutToNonGradInInfo(concat_grad_op.gradOutToNonGradInInfo) {}

std::unique_ptr<Op> ConcatGradOp::clone() const {
  return std::make_unique<ConcatGradOp>(*this);
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

void ConcatGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("start", start);
  os.appendAttribute("end", end);
}

int64_t ConcatGradOp::getAxis() const { return axis; }

int64_t ConcatGradOp::getStart() const { return start; }

int64_t ConcatGradOp::getEnd() const { return end; }

namespace {

static OpCreator<ConcatOp> concatOpCreator(
    {Onnx::Operators::Concat_1, Onnx::Operators::Concat_4},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis");

      return std::unique_ptr<Op>(new ConcatOp(_opid, axis, settings));
    },
    true);

} // namespace

} // namespace poponnx
