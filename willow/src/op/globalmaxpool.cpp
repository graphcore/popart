#include <popart/error.hpp>
#include <popart/op/globalmaxpool.hpp>

#include <memory>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

GlobalMaxPoolOp::GlobalMaxPoolOp(const OperatorIdentifier &_opid,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void GlobalMaxPoolOp::setup() {

  // If the input is N x C x D1 x D2 then the output shape N x C x 1 x 1
  // If the input is N x C x D1 ... Dn then the output shape N x C x 1 x ... x 1
  auto gmp = inShape(getInIndex());
  std::fill(gmp.begin() + 2, gmp.end(), 1);
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), gmp};

  kernel =
      Shape(inShape(getInIndex()).begin() + 2, inShape(getInIndex()).end());
}

void GlobalMaxPoolOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("kernel", kernel);
}

Shape GlobalMaxPoolOp::getStrides() const {
  Shape strides(kernel.size());
  std::fill(strides.begin(), strides.end(), 1);
  return strides;
}
Shape GlobalMaxPoolOp::getLowerPads() const {
  Shape lowerPads(kernel.size());
  std::fill(lowerPads.begin(), lowerPads.end(), 0);
  return lowerPads;
}
Shape GlobalMaxPoolOp::getUpperPads() const {
  Shape lowerPads(kernel.size());
  std::fill(lowerPads.begin(), lowerPads.end(), 0);
  return lowerPads;
}

std::unique_ptr<Op> GlobalMaxPoolOp::clone() const {
  return std::make_unique<GlobalMaxPoolOp>(*this);
}

std::vector<std::unique_ptr<Op>> GlobalMaxPoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<GlobalMaxPoolGradOp>(*this));
  return upops;
}

GlobalMaxPoolGradOp::GlobalMaxPoolGradOp(const GlobalMaxPoolOp &op_)
    : Op(Onnx::GradOperators::GlobalMaxPoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(GlobalMaxPoolOp::getInIndex())),
      cloneOfCreator(op_.clone()) {}

void GlobalMaxPoolGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendForwardOp(getCloneOfCreator());
}

const std::vector<GradInOutMapper> &GlobalMaxPoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the max pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the max pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledInIndex(),
       GlobalMaxPoolOp::getOutIndex(),
       GradOpInType::GRADOUT},
      {getPooledInIndex(), GlobalMaxPoolOp::getOutIndex(), GradOpInType::OUT},
      {getPrePooledInIndex(), GlobalMaxPoolOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

// The input to the max pool (PrePooled) is
// the input to the grad op at index 0.

const std::map<int, int> &GlobalMaxPoolGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), GlobalMaxPoolOp::getInIndex()}};
  return outInfo;
}

void GlobalMaxPoolGradOp::setup() { outInfo(getOutIndex()) = unpooledInfo; }

std::unique_ptr<Op> GlobalMaxPoolGradOp::clone() const {
  return std::make_unique<GlobalMaxPoolGradOp>(*this);
}

const GlobalMaxPoolOp *GlobalMaxPoolGradOp::getCloneOfCreator() const {
  return dynamic_cast<GlobalMaxPoolOp *>(cloneOfCreator.get());
}

namespace {
static OpCreator<GlobalMaxPoolOp>
    globalMaxPoolOpCreator({Onnx::Operators::GlobalMaxPool_1});
} // namespace

} // namespace popart
