#include <algorithm>
#include <memory>
#include <popart/op/transpose.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

TransposeBaseOp::TransposeBaseOp(const OperatorIdentifier &_opid,
                                 const std::vector<int64_t> &perm_,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_), perm(perm_) {}

view::RegMap TransposeBaseOp::fwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in TransposeBaseOp::fwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  // being conservative and returning the full region,
  // even for non-full input region :
  auto outRegion   = view::Region::getFull(outInfo(getOutIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(outRank(getOutIndex()));
  return [emptyRegion, outRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return emptyRegion;
    }
    return outRegion;
  };
}

view::RegMap TransposeBaseOp::bwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in TransposeBaseOp::bwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  auto inRegion    = view::Region::getFull(inInfo(getInIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(inRank(getInIndex()));
  return [emptyRegion, inRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return emptyRegion;
    }
    return inRegion;
  };
}

TransposeOp::TransposeOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &perm_,
                         const Op::Settings &settings_)
    : TransposeBaseOp(_opid, perm_, settings_) {}

std::unique_ptr<Op> TransposeOp::clone() const {
  return std::make_unique<TransposeOp>(*this);
}

std::vector<std::unique_ptr<Op>> TransposeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<TransposeGradOp>(*this));
  return upops;
}

void TransposeBaseOp::setup() {
  auto in_shape = inInfo(getInIndex()).shape();

  // If perm is empty, set the the default value
  if (perm.empty()) {
    setDefaultPerm();
  }

  Shape out_shape;
  for (auto i : perm) {
    out_shape.push_back(in_shape[i]);
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).data_type(), out_shape};
}

std::vector<int64_t> TransposeBaseOp::generateReversePermutation() const {
  std::vector<int64_t> reverse_perm(perm.size());
  for (int i = 0; i < perm.size(); i++) {
    reverse_perm[perm[i]] = i;
  }

  return reverse_perm;
}

void TransposeBaseOp::setDefaultPerm() {
  auto in_shape = inInfo(getInIndex()).shape();

  // default behaviour is to reverse the shape of the input tensor
  if (perm.empty()) {
    for (int64_t i = in_shape.size() - 1; i >= 0; i--) {
      perm.push_back(i);
    }
  }
}

void TransposeOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("perm", getPerm());
}

bool TransposeOp::canBeReplacedByIdentity() {
  return std::is_sorted(getPerm().begin(), getPerm().end());
}

std::unique_ptr<Op>
TransposeOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::TransposeInplace) {
    return std::make_unique<TransposeInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}
std::vector<std::tuple<OperatorIdentifier, float>>
TransposeOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::TransposeInplace, 10}};
}

TransposeGradOp::TransposeGradOp(const TransposeOp &fwdOp)
    : TransposeOp(Onnx::GradOperators::TransposeGrad,
                  fwdOp.generateReversePermutation(),
                  fwdOp.getSettings()) {}

std::unique_ptr<Op> TransposeGradOp::clone() const {
  return std::make_unique<TransposeGradOp>(*this);
}

const std::vector<GradInOutMapper> &TransposeGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), TransposeOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &TransposeGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), TransposeOp::getInIndex()}};

  return outInfo;
}

TransposeInplaceOp::TransposeInplaceOp(const TransposeOp &op)
    : TransposeBaseOp(Onnx::CustomOperators::TransposeInplace,
                      op.getPerm(),
                      op.settings) {}

std::unique_ptr<Op> TransposeInplaceOp::clone() const {
  return std::make_unique<TransposeInplaceOp>(*this);
}

namespace {
static OpCreator<TransposeOp> transposeOpCreator(
    Onnx::Operators::Transpose_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      Shape perm = attr.getAttribute<Attributes::Ints>("perm", {});
      return std::unique_ptr<Op>(new TransposeOp(_opid, perm, settings));
    },
    true);
} // namespace

} // namespace popart
