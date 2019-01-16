#include <poponnx/makeunique.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

bool ReluOp::hasInplaceVariant(InIndex) const {
  // we could throw an error if the indices are not both zero,
  // but we assume that both the in and out indices are zero.
  // In which case, the ReluOp does have an inplace variant.
  return true;
}

ReluInplaceOp::ReluInplaceOp(ReluOp *relu_op)
    : Op(Onnx::CustomOperators::ReluInplace, relu_op->pir) {}

void ReluInplaceOp::setup() {
  // no output, nothing to setup
  outInfo(ReluOp::getOutIndex()) = inInfo(ReluOp::getInIndex());
}

// we do not check that the InIndex is 0, we could
bool ReluInplaceOp::modifies(InIndex) const { return true; }

std::unique_ptr<Op> ReluOp::clone() const { return make_unique<ReluOp>(*this); }

std::unique_ptr<Op> ReluOp::getInplaceVariant(InIndex) {
  return make_unique<ReluInplaceOp>(this);
}

ReluOp::ReluOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name,
               const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::vector<std::unique_ptr<Op>> ReluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ReluGradOp>(this));
  return upops;
}

void ReluOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

void ReluGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradReludInIndex());
}

ReluGradOp::ReluGradOp(ReluOp *op_)
    : Op(Onnx::GradOperators::ReluGrad, op_->pir) {}

const std::vector<GradInOutMapper> &ReluGradOp::gradInputInfo() const {
  // input at index getGradReludIn() (=0) : gradient of output of relu
  // input at index getReludIn() (=1)     : output of relu
  // can we do better sometimes with in-placing?
  // The 0's below : As there is only 1 output of Relu, it
  // is output at index 0.
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradReludInIndex(), ReluOp::getOutIndex(), GradOpInType::GRADOUT},
      {getReludInIndex(), ReluOp::getOutIndex(), GradOpInType::OUT}};
  return inInfo;
}

const std::map<int, int> &ReluGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReluOp::getInIndex()}};
  return outInfo;
}

namespace {
static OpCreator<ReluOp> reluOpCreator(Onnx::Operators::Relu_6);
static GradOpCreator<ReluInplaceOp>
    reluInPlaceOpCreator(Onnx::CustomOperators::ReluInplace);
static GradOpCreator<ReluGradOp>
    reluGradOpCreator(Onnx::GradOperators::ReluGrad);
} // namespace

} // namespace poponnx
