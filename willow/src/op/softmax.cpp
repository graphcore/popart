#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SoftmaxOp::SoftmaxOp(const OperatorIdentifier &_opid,
                     int64_t axis_,
                     const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

std::vector<std::unique_ptr<Op>> SoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SoftmaxGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> SoftmaxOp::clone() const {
  return make_unique<SoftmaxOp>(*this);
}

int64_t SoftmaxOp::getAxis() const { return axis; }

void SoftmaxOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

void SoftmaxGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradProbsInIndex());
}

SoftmaxGradOp::SoftmaxGradOp(const SoftmaxOp &op_)
    : Op(Onnx::GradOperators::SoftmaxGrad, op_.getSettings()),
      axis(op_.getAxis()) {}

const std::vector<GradInOutMapper> &SoftmaxGradOp::gradInputInfo() const {
  // input at index 0 (probGradInputIndex()) : gradient of output of softmax
  // input at index 1 (actsIn()): input of softmax (activations before p)
  // the (1-sparse) gradient of the output will be used
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradProbsInIndex(), SoftmaxOp::getOutIndex(), GradOpInType::GRADOUT},
      {getActsInIndex(), SoftmaxOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &SoftmaxGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SoftmaxOp::getInIndex()}};
  return outInfo;
}

int64_t SoftmaxGradOp::getAxis() const { return axis; }

void SoftmaxGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

SoftmaxGradDirectOp::SoftmaxGradDirectOp(const NllLoss *nls,
                                         const Op::Settings &settings_)
    : Op(Onnx::CustomGradOperators::SoftmaxGradDirect, settings_) {
  nllloss_ = nls;
}

const NllLoss *SoftmaxGradDirectOp::nlll() const { return nllloss_; }

std::unique_ptr<Op> SoftmaxGradDirectOp::clone() const {
  throw error("Unexpected (but valid) request to clone SoftmaxGradDirectOp");
}

void SoftmaxGradDirectOp::setup() {
  // gradient of activations has same shape as probabilities
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

namespace {
static OpCreator<SoftmaxOp> softmaxOpCreator(
    Onnx::Operators::Softmax_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis", 1);

      return std::unique_ptr<Op>(new SoftmaxOp(_opid, axis, settings));
    },
    true);

// static GradOpCreator<SoftmaxGradOp>
//    softmaxGradOpCreator(Onnx::GradOperators::SoftmaxGrad);
// static GradOpCreator<SoftmaxGradDirectOp>
//    softmaxGradDirectOpCreator(Onnx::CustomGradOperators::SoftmaxGradDirect);
} // namespace

} // namespace poponnx
