// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <memory>
#include <popart/graph.hpp>
#include <popart/op/concat.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

ConcatInplaceOp::ConcatInplaceOp(int64_t axis_, const Op::Settings &settings_)
    : ConcatOp(Onnx::CustomOperators::ConcatInplace, axis_, settings_) {}

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
  if (inIndex >= input->tensorMap().size() || inIndex < 0) {
    throw error("invalid index in ConcatOp::fwdRegMap");
  }
}

view::RegMap ConcatOp::fwdRegMap(InIndex inIndex, OutIndex) const {
  regMapPreChecks(inIndex);
  int64_t offset = getOutOffset(inIndex);
  int64_t axisIn = getAxis();
  return [axisIn, offset](const view::Region &r_in) {
    view::LowBounds lower = r_in.getLower();
    view::UppBounds upper = r_in.getUpper();
    lower[axisIn] += offset;
    upper[axisIn] += offset;
    return view::Regions(1, view::Region(lower, upper));
  };
}

view::RegMap ConcatOp::bwdRegMap(InIndex inIndex, OutIndex) const {
  regMapPreChecks(inIndex);
  int64_t offset = getOutOffset(inIndex);
  int64_t axisIn = getAxis();
  return [axisIn, offset](const view::Region &r_out) {
    view::LowBounds lower = r_out.getLower();
    view::UppBounds upper = r_out.getUpper();
    lower[axisIn] -= offset;
    upper[axisIn] -= offset;
    // TODO T8446 : check intersect?
    return view::Regions(1, view::Region(lower, upper));
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

int64_t ConcatOp::getAxis() const {
  // Onnx 11 supports negative axis indexing for argmin and argmax.
  if (axis >= 0) {
    return axis;
  } else {
    return inInfo(getInIndex(0)).rank() + axis;
  }
}

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
          "Input {} has rank {} ({}) while the output has rank {} ({})."
          "Concat axis {}.",
          i,
          inputs.size(),
          i,
          shape.size(),
          shape,
          outShape.size(),
          outShape,
          axis);
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

void ConcatOp::validateAxis() const {
  auto r = static_cast<int64_t>(inShape(InIndex(0)).size());
  if (axis < -r || axis > r - 1) {
    throw error("Attribute 'axis' of ConcatOp, {}, is outside of acceptable "
                "range [{}, {}]",
                -r,
                r - 1);
  }
}

void ConcatOp::setup() {
  validateAxis();

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
    outShape = getOutputShape(getAxis(), inputs);
  } catch (const error &) {
    logging::op::err(
        "Error trying to calculate output shape for concat {}({}, {})",
        id,
        opid,
        name());
    throw;
  }

  outInfo(getOutIndex()) = TensorInfo(outType, outShape);
}

int64_t ConcatOp::getOutOffset(int64_t dim) const {
  int64_t offset = 0;
  for (int i = 0; i < dim; i++) {
    const auto shape = inShape(getInIndex(i));
    offset += shape.at(getAxis());
  }
  return offset;
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
bool ConcatOp::canBeReplacedByIdentity() const { return input->n() == 1; }

ConcatGradOp::ConcatGradOp(const ConcatOp &fwd, InIndex inputIndex)
    : Op(Onnx::GradOperators::ConcatGrad, fwd.getSettings()),
      axis(fwd.getAxis()), start(0), end(0), fwdInput(inputIndex) {
  for (int i = 0; i < inputIndex; ++i) {
    auto shape = fwd.inShape(ConcatOp::getInIndex(i));
    start += shape[getAxis()];
  }

  for (int i = 0; i <= inputIndex; ++i) {
    auto shape = fwd.inShape(ConcatOp::getInIndex(i));
    end += shape[getAxis()];
  }

  const DataType outType =
      fwd.inInfo(ConcatOp::getInIndex(fwdInput)).dataType();
  Shape outShape      = fwd.inShape(ConcatOp::getInIndex(fwdInput));
  outShape[getAxis()] = end - start;

  gradInfo               = TensorInfo(outType, outShape);
  gradOutToNonGradInInfo = {{getOutIndex(), ConcatOp::getInIndex(fwdInput)}};
}

void ConcatOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
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
      {getInIndex(), ConcatOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &ConcatGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

void ConcatGradOp::setup() { outInfo(getOutIndex()) = gradInfo; }

void ConcatGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("start", start);
  os.appendAttribute("end", end);
}

int64_t ConcatGradOp::getAxis() const { return axis; }

int64_t ConcatGradOp::getStart() const { return start; }

int64_t ConcatGradOp::getEnd() const { return end; }

void ConcatGradOp::configureShardedOp(Op *const shardedOp,
                                      const Settings *const settings_) const {
  Op::configureShardedOp(shardedOp, settings_);

  auto oldInShape = inInfo(ConcatGradOp::getInIndex()).shape();

  Tensor *inTensor  = shardedOp->input->tensor(ConcatGradOp::getInIndex());
  Tensor *outTensor = shardedOp->output->tensor(ConcatGradOp::getOutIndex());
  auto newInShape   = inTensor->info.shape();
  int64_t reduction = 1;
  for (size_t i = 0; i < oldInShape.size(); ++i) {
    reduction = oldInShape.at(i) / newInShape.at(i);
    if (reduction > 1) {
      Op *prod                     = outTensor->getProducer();
      ConcatGradOp *concatGradProd = dynamic_cast<ConcatGradOp *>(prod);
      auto type                    = concatGradProd->gradInfo.dataType();
      auto shape                   = concatGradProd->gradInfo.shape();
      shape[i] /= reduction;
      concatGradProd->gradInfo.set(type, shape);
      concatGradProd->setup();
      break;
    }
  }
}

namespace {

// Do we support other types??
static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition concatOpDef({OpDefinition::Inputs({{"inputs", T}}),
                                 OpDefinition::Outputs({{"concat_result", T}}),
                                 OpDefinition::Attributes({
                                     {"axis", {"*"}},
                                 })});

static OpCreator<ConcatOp> concatOpCreator(
    OpDefinitions({{Onnx::Operators::Concat_1, concatOpDef},
                   {Onnx::Operators::Concat_4, concatOpDef},
                   {Onnx::Operators::Concat_11, concatOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis");

      return std::unique_ptr<Op>(new ConcatOp(info.opid, axis, info.settings));
    },
    true);
} // namespace

} // namespace popart
