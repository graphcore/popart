#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

SoftmaxOp::SoftmaxOp(const OperatorIdentifier &_opid,
                     int64_t axis_,
                     const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

SoftmaxInplaceOp::SoftmaxInplaceOp(const SoftmaxOp &softmax_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SoftmaxInplace,
                                softmax_op.getSettings()),
      axis(softmax_op.getAxis()) {}

std::vector<std::unique_ptr<Op>> SoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SoftmaxGradOp>(*this));
  return upops;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SoftmaxOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SoftmaxInplace, 10}};
}

std::unique_ptr<Op> SoftmaxInplaceOp::clone() const {
  return std::make_unique<SoftmaxInplaceOp>(*this);
}

std::unique_ptr<Op>
SoftmaxOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SoftmaxInplace) {
    return std::make_unique<SoftmaxInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::unique_ptr<Op> SoftmaxOp::clone() const {
  return std::make_unique<SoftmaxOp>(*this);
}

int64_t SoftmaxOp::getAxis() const {
  if (axis < 0) {
    return inRank(getInIndex()) + axis;
  } else {
    return axis;
  }
}

void SoftmaxOp::setAxis(int64_t axis_) { axis = axis_; }

void SoftmaxOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

void SoftmaxInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

void SoftmaxGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradProbsInIndex());
}

SoftmaxGradOp::SoftmaxGradOp(const SoftmaxOp &op_)
    : Op(Onnx::GradOperators::SoftmaxGrad, op_.getSettings()),
      axis(op_.getAxis()) {}

std::unique_ptr<Op> SoftmaxGradOp::clone() const {
  return std::make_unique<SoftmaxGradOp>(*this);
}

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

void SoftmaxGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
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

  // When replacing a SoftmaxGradOp with a SoftmaxGradDirectOp, we are
  // disconnecting the path from the final loss to the ops downstream.
  // However, we don't want updateVertices() alter the attributes of these
  // ops after performing the SoftmaxGradDirectOp transform. So here we
  // 'pretend' that the SoftmaxGradDirectOp has a path from the final loss.
  fromLoss = PathFromLoss::Yes;
}

bool SoftmaxGradDirectOp::hasNlllFwdOp() const {
  TensorId lossTensorName = nlll()->output(NllLoss::getOutIndex());
  if (getGraph().getTensors().contains(lossTensorName)) {
    auto t = getGraph().getTensors().get(lossTensorName);
    return t->hasProducer();
  }
  return false;
}

Op *SoftmaxGradDirectOp::nlllFwdOp() const {

  // First check that the forward Nll loss op exists
  // in the ir
  if (!hasNlllFwdOp()) {
    throw error("The forward loss op corresponding to the SoftmaxGradDirectOp "
                "{} does not exist in the Ir",
                id);
  }

  TensorId lossTensorName = nlll()->output(NllLoss::getOutIndex());

  // Find the op producing the loss tensor, i.e. the
  // corresponding fwd loss op whose bwd op has merged
  // with the SoftmaxGradOp
  Tensor *lossTensor = getGraph().getTensors().get(lossTensorName);
  Op *fwdLossOp      = lossTensor->getProducer();

  return fwdLossOp;
}

NlllWithSoftmaxGradDirectOp::NlllWithSoftmaxGradDirectOp(
    const NllLoss *nls,
    const Op::Settings &settings_)
    : Op(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect, settings_) {
  nllloss_ = nls;
}

const NllLoss *NlllWithSoftmaxGradDirectOp::nlll() const { return nllloss_; }

std::unique_ptr<Op> NlllWithSoftmaxGradDirectOp::clone() const {
  throw error(
      "Unexpected (but valid) request to clone NlllWithSoftmaxGradDirectOp");
}

void NlllWithSoftmaxGradDirectOp::setup() {

  // connect the loss scaling tensor if is non-const
  if (!getIr().getOptimizer().lossScaling().isConst()) {
    connectInTensor(NlllWithSoftmaxGradDirectOp::getLossScalingInIndex(),
                    getIr().getOptimizer().getLossScalingTensorId(
                        inInfo(nlll()->getProbsInIndex()).dataType()));
  }

  // gradient of activations has same shape as probabilities
  outInfo(getGradOutIndex()) = inInfo(getProbsInIndex());

  // Outputs a loss for each label index.
  // Same shape as label input, same datatype as probs input
  outInfo(getLossOutIndex())
      .set(inInfo(getProbsInIndex()).dataType(),
           inInfo(getLabelInIndex()).shape());

  // When replacing a Nlll and SoftmaxGradOp with a NlllWithSoftmaxGradDirectOp,
  // we are disconnecting the path from the final loss to the ops downstream.
  // However, we don't want updateVertices() alter the attributes of these
  // ops after performing the NlllWithSoftmaxGradDirectOp transform. So here we
  // 'pretend' that the NlllWithSoftmaxGradDirectOp has a path from the final
  // loss.
  fromLoss = PathFromLoss::Yes;
}

namespace {
static OpCreator<SoftmaxOp> softmaxOpCreator(
    {Onnx::Operators::Softmax_1, Onnx::Operators::Softmax_11},
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

} // namespace popart
