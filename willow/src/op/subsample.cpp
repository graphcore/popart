#include <algorithm>
#include <onnx/defs/schema.h>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/subsample.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

view::RegMap SubsampleBaseOp::fwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in SubsampleBaseOp::fwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  // being conservative and returning the full region,
  // even for non-full input region :
  auto outRegion = view::Region::getFull(outInfo(getOutIndex()).shape());
  return [outRegion](const view::Region &) { return outRegion; };
}

view::RegMap SubsampleBaseOp::bwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in SubsampleBaseOp::bwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  auto inRegion = view::Region::getFull(inInfo(getInIndex()).shape());
  return [inRegion](const view::Region &) { return inRegion; };
}

SubsampleBaseOp::SubsampleBaseOp(const OperatorIdentifier &_opid,
                                 const std::vector<int64_t> &strides_,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_), strides(strides_) {}

std::unique_ptr<Op> SubsampleOp::clone() const {
  return make_unique<SubsampleOp>(*this);
}

std::vector<std::unique_ptr<Op>> SubsampleBaseOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SubsampleGradOp>(*this));
  return upops;
}

// We are subsampling the tensor
void SubsampleBaseOp::setup() {

  // Get the stride attribute
  // nAtts.set(strides, "strides");

  // Verify that a stride of 0 has not be used
  for (int i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0)
      throw error("Strides invalid. 0 stride at index {}", i);
  }

  // Type will be the same
  DataType outType = inInfo(getInIndex()).dataType();

  // Now calculate the shape of the output tensor.
  // The rank will be the same, but the value of the dimensions will be
  // different
  Shape outShape;
  Shape _inShape = inShape(getInIndex());
  for (int i = 0; i < strides.size(); ++i) {
    // We have already checked for a stride of 0, so we do not have to worry
    // about divide by 0 Poplar rounds up if stride is not an a factor of the
    // dimension
    outShape.push_back((_inShape[i] + strides[i] - 1) / strides[i]);
  }

  outInfo(getOutIndex()).set(outType, outShape);
}

std::vector<uint32_t> SubsampleBaseOp::strides_u32() const {
  return vXtoY<int64_t, uint32_t>(strides);
}

bool SubsampleBaseOp::strideSizeOne() const {
  return std::all_of(
      strides.cbegin(), strides.cend(), [](int64_t p) { return p == 1; });
}

SubsampleInplaceOp::SubsampleInplaceOp(const SubsampleOp &op)
    : SubsampleBaseOp(Onnx::CustomOperators::SubsampleInplace,
                      op.getStrides(),
                      op.settings) {}

view::Region SubsampleInplaceOp::aliases(InIndex index) const {
  return uses(index);
}

std::unique_ptr<Op> SubsampleInplaceOp::clone() const {
  return make_unique<SubsampleInplaceOp>(*this);
}

std::unique_ptr<Op>
SubsampleOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SubsampleInplace) {
    return make_unique<SubsampleInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

SubsampleOp::SubsampleOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &strides_,
                         const Op::Settings &settings_)
    : SubsampleBaseOp(_opid, strides_, settings_) {}

// A subsample with all strides being  1 can be replaced by identity
bool SubsampleBaseOp::canBeReplacedByIdentity() { return strideSizeOne(); }

void SubsampleBaseOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("strides", strides);
}

SubsampleGradOp::SubsampleGradOp(const SubsampleBaseOp &fwdOp_)
    : Op(Onnx::CustomGradOperators::SubsampleGrad, fwdOp_.getSettings()),
      strides(fwdOp_.getStrides()), fwdOpInfo(fwdOp_.inInfo(0)) {}

std::unique_ptr<Op> SubsampleGradOp::clone() const {
  return make_unique<SubsampleGradOp>(*this);
}

void SubsampleGradOp::setup() { output->tensor(0)->info = fwdOpInfo; }

const std::vector<GradInOutMapper> &SubsampleGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SubsampleBaseOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &SubsampleGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SubsampleBaseOp::getInIndex()}};

  return outInfo;
}

std::vector<uint32_t> SubsampleGradOp::strides_u32() const {
  return vXtoY<int64_t, uint32_t>(strides);
}

void SubsampleGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("strides", strides);
}

namespace {
static OpCreator<SubsampleOp> subsampleOpCreator(
    Onnx::CustomOperators::Subsample_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> strides =
          attr.getAttribute<Attributes::Ints>("strides", {});

      return std::unique_ptr<Op>(new SubsampleOp(_opid, strides, settings));
    },
    true);
} // namespace

} // namespace poponnx
