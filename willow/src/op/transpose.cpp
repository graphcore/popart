#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/transpose.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TransposeOp::TransposeOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &perm_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), perm(perm_) {

  //  nAtts.setIfPresent(perm, "perm");
}

std::unique_ptr<Op> TransposeOp::clone() const {
  return make_unique<TransposeOp>(*this);
}

std::vector<std::unique_ptr<Op>> TransposeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<TransposeGradOp>(*this));
  return upops;
}

void TransposeOp::setup() {
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

const std::vector<int64_t> &TransposeOp::getPerm() const { return perm; }

std::vector<int64_t> TransposeOp::generateReversePermutation() const {
  std::vector<int64_t> reverse_perm(perm.size());
  for (int i = 0; i < perm.size(); i++) {
    reverse_perm[perm[i]] = i;
  }

  return reverse_perm;
}

void TransposeOp::setDefaultPerm() {
  auto in_shape = inInfo(getInIndex()).shape();

  // default behaviour is to reverse the shape of the input tensor
  if (perm.empty()) {
    for (int64_t i = in_shape.size() - 1; i >= 0; i--) {
      perm.push_back(i);
    }
  }
}

void TransposeOp::appendAttributes(std::stringstream &ss,
                                   const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "perm", perm);
}

TransposeGradOp::TransposeGradOp(const TransposeOp &fwdOp)
    : TransposeOp(Onnx::GradOperators::TransposeGrad,
                  fwdOp.generateReversePermutation(),
                  fwdOp.getSettings()) {}

std::unique_ptr<Op> TransposeGradOp::clone() const {
  return make_unique<TransposeGradOp>(*this);
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

bool TransposeOp::canBeReplacedByIdentity() {
  return std::is_sorted(perm.begin(), perm.end());
}

namespace {
static OpCreator<TransposeOp> transposeOpCreator(
    Onnx::Operators::Transpose_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> perm =
          attr.getAttribute<Attributes::Ints>("perm", {});

      return std::unique_ptr<Op>(new TransposeOp(_opid, perm, settings));
    },
    true);
} // namespace

} // namespace poponnx
