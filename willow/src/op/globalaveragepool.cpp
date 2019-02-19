#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/globalaveragepool.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GlobalAveragePoolOp::GlobalAveragePoolOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void GlobalAveragePoolOp::setup() {
  // If the input is N x C x D1 x D2 then the output shape N x C x 1 x 1
  // If the input is N x C x D1 ... Dn then the output shape N x C x 1 x 1
  Shape gapShape = {inShape(getInIndex())[0], inShape(getInIndex())[1], 1, 1};
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), gapShape};

  kernel =
      Shape(inShape(getInIndex()).begin() + 2, inShape(getInIndex()).end());
}

Shape GlobalAveragePoolOp::getStrides() const {
  Shape strides(kernel.size());
  std::fill(strides.begin(), strides.end(), 1);
  return strides;
}
Shape GlobalAveragePoolOp::getLowerPads() const {
  Shape lowerPads(kernel.size());
  std::fill(lowerPads.begin(), lowerPads.end(), 0);
  return lowerPads;
}
Shape GlobalAveragePoolOp::getUpperPads() const {
  Shape lowerPads(kernel.size());
  std::fill(lowerPads.begin(), lowerPads.end(), 0);
  return lowerPads;
}

std::unique_ptr<Op> GlobalAveragePoolOp::clone() const {
  return make_unique<GlobalAveragePoolOp>(*this);
}

std::vector<std::unique_ptr<Op>> GlobalAveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<GlobalAveragePoolGradOp>(*this));
  return upops;
}

GlobalAveragePoolGradOp::GlobalAveragePoolGradOp(const GlobalAveragePoolOp &op_)
    : Op(Onnx::GradOperators::GlobalAveragePoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(GlobalAveragePoolOp::getInIndex())),
      cloneOfCreator(op_.clone()) {}

const std::vector<GradInOutMapper> &
GlobalAveragePoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the average pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the average pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledInIndex(),
       GlobalAveragePoolOp::getOutIndex(),
       GradOpInType::GRADOUT},
      {getPooledInIndex(),
       GlobalAveragePoolOp::getOutIndex(),
       GradOpInType::OUT},
      {getPrePooledInIndex(),
       GlobalAveragePoolOp::getInIndex(),
       GradOpInType::IN}};
  return inInfo;
}

// The input to the average pool (PrePooled) is
// the input to the grad op at index 0.

const std::map<int, int> &GlobalAveragePoolGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), GlobalAveragePoolOp::getInIndex()}};
  return outInfo;
}

void GlobalAveragePoolGradOp::setup() { outInfo(getOutIndex()) = unpooledInfo; }

const GlobalAveragePoolOp *GlobalAveragePoolGradOp::getCloneOfCreator() {
  return dynamic_cast<GlobalAveragePoolOp *>(cloneOfCreator.get());
}

namespace {
static OpCreator<GlobalAveragePoolOp>
    globalAveragePoolOpCreator({Onnx::Operators::GlobalAveragePool_1});
} // namespace

} // namespace poponnx
