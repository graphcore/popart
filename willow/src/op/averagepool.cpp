#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

AveragePoolOp::AveragePoolOp(const OperatorIdentifier &_opid,
                             Ir *_ir,
                             const std::string &name,
                             const Attributes &_attr)
    : HasReceptiveFieldOp(_opid, _ir, name, _attr) {}

void AveragePoolOp::setup0() {}

void AveragePoolOp::setSpatialK() {
  spatialK.resize(nSpatialDims);
  std::vector<int64_t> kernel_shape;
  nAtts.setIfPresent(kernel_shape, "kernel_shape");
  if (kernel_shape.size() != inRank(getInIndex()) - 2) {
    throw error(
        "invalid kernel_shape, not same rank as the tensor operated on");
  }
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatialK[spDim] = kernel_shape[spDim];
  }
}

const AveragePoolOp *AveragePoolGradOp::getCloneOfCreator() {
  return dynamic_cast<AveragePoolOp *>(cloneOfCreator.get());
}

std::unique_ptr<Op> AveragePoolOp::clone() const {
  return make_unique<AveragePoolOp>(*this);
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> AveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<AveragePoolGradOp>(this));
  return upops;
}

AveragePoolGradOp::AveragePoolGradOp(AveragePoolOp *op_)
    : Op({Onnx::GradOperators::AveragePoolGrad, op_->pir, {}}),
      unpooledInfo(op_->inInfo(AveragePoolOp::getInIndex())),
      cloneOfCreator(op_->clone()) {}

const std::vector<GradInOutMapper> &AveragePoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the average pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the average pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledInIndex(),
       AveragePoolOp::getOutIndex(),
       GradOpInType::GRADOUT},
      {getPooledInIndex(), AveragePoolOp::getOutIndex(), GradOpInType::OUT},
      {getPrePooledInIndex(), AveragePoolOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

// The input to the average pool (PrePooled) is
// the input to the grad op at index 0.

const std::map<int, int> &AveragePoolGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AveragePoolOp::getInIndex()}};
  return outInfo;
}

void AveragePoolGradOp::setup() { outInfo(getOutIndex()) = unpooledInfo; }

namespace {
static OpCreator<AveragePoolOp>
    averagePoolOpCreator(Onnx::Operators::AveragePool);
static GradOpCreator<AveragePoolGradOp>
    averagePoolGradOpCreator(Onnx::GradOperators::AveragePoolGrad);
} // namespace

} // namespace poponnx
