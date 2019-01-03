#include <algorithm>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

ConcatOp::ConcatOp(const OperatorIdentifier &_opid,
                   Ir *_ir,
                   const std::string &name,
                   const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  _attr.setIfPresent(axis, "axis");
}

ConcatOp::ConcatOp(const OperatorIdentifier &_opid, ConcatOp *concat_op)
    : Op(_opid, concat_op->pir), axis(concat_op->getAxis()) {}

ConcatInplaceOp::ConcatInplaceOp(ConcatOp *concat_op)
    : ConcatOp(Onnx::CustomOperators::ConcatInplace, concat_op->pir) {}

std::unique_ptr<Op> ConcatOp::clone() const {
  return make_unique<ConcatOp>(*this);
}

std::unique_ptr<Op> ConcatInplaceOp::clone() const {
  return make_unique<ConcatInplaceOp>(*this);
}

int64_t ConcatOp::getAxis() const { return axis; }

bool ConcatOp::hasInplaceVariant(InIndex index) const {
  const std::vector<InIndex> inIndices = {index};

  return hasInplaceVariant(inIndices);
}

bool ConcatOp::hasInplaceVariant(const std::vector<InIndex> &inIndices) const {
  if (inIndices.size() != input->n()) {
    return false;
  }

  for (auto index : inIndices) {
    if (!input->hasIndex(index)) {
      return false;
    }
  }

  return true;
}

bool ConcatInplaceOp::hasInplaceVariant(InIndex) const { return false; }
bool ConcatInplaceOp::hasInplaceVariant(const std::vector<InIndex> &) const {
  return false;
}

std::unique_ptr<Op> ConcatOp::getInplaceVariant(InIndex index) {
  const std::vector<InIndex> inIndices = {index};

  return getInplaceVariant(inIndices);
}

std::unique_ptr<Op>
ConcatOp::getInplaceVariant(const std::vector<InIndex> &inIndices) {
  if (!hasInplaceVariant(inIndices)) {
    throw error("ConcatOp::getInplaceVariant : Given indices cannot be used to "
                "create the inplace op");
  }

  return make_unique<ConcatInplaceOp>(this);
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
    result.push_back(make_unique<ConcatGradOp>(this, i));
  }

  return result;
}

ConcatGradOp::ConcatGradOp(ConcatOp *fwd, InIndex input)
    : Op({Onnx::GradOperators::ConcatGrad, fwd->pir, {}}), axis(fwd->getAxis()),
      start(0), end(0), fwdInput(input) {
  for (int i = 0; i < input; ++i) {
    auto shape = fwd->inShape(ConcatOp::getInIndex(i));
    start += shape[axis];
  }

  for (int i = 0; i <= input; ++i) {
    auto shape = fwd->inShape(ConcatOp::getInIndex(i));
    end += shape[axis];
  }

  const DataType outType =
      fwd->inInfo(ConcatOp::getInIndex(fwdInput)).dataType();
  Shape outShape = fwd->inShape(ConcatOp::getInIndex(fwdInput));
  outShape[axis] = end - start;

  gradInfo               = TensorInfo(outType, outShape);
  gradOutToNonGradInInfo = {{getOutIndex(), ConcatOp::getInIndex(fwdInput)}};
}

ConcatGradOp::ConcatGradOp(const OperatorIdentifier &_opid,
                           ConcatGradOp *concat_grad_op)
    : Op({_opid, concat_grad_op->pir, {}}), axis(concat_grad_op->axis),
      start(concat_grad_op->start), end(concat_grad_op->end),
      fwdInput(concat_grad_op->fwdInput), gradInfo(concat_grad_op->gradInfo),
      gradOutToNonGradInInfo(concat_grad_op->gradOutToNonGradInInfo) {}

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

static OpCreator<ConcatOp>
    concatOpCreator(Onnx::Operators::Concat,
                    [](const OperatorIdentifier &opid,
                       Ir *ir,
                       const std::string &name = "",
                       const Attributes &attr  = {}) -> std::unique_ptr<Op> {
                      return make_unique<ConcatOp>(opid, ir, name, attr);
                    },
                    true);

static GradOpCreator<ConcatInplaceOp>
    concatInplaceOpCreator(Onnx::CustomOperators::ConcatInplace);
static GradOpCreator<ConcatGradOp>
    concatArgGradOpCreator(Onnx::GradOperators::ConcatGrad);
} // namespace

} // namespace poponnx
