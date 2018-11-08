#include <willow/averagepool.hpp>
#include <willow/error.hpp>
#include <willow/tensor.hpp>

namespace willow {

AveragePoolOp::AveragePoolOp(const onnx::NodeProto &node, Ir *pir)
    : HasReceptiveFieldOp(node, pir) {}

void AveragePoolOp::setup0() {}

void AveragePoolOp::setSpatialK() {
  spatialK.resize(nSpatialDims);
  std::vector<int64_t> kernel_shape;
  nAtts.setIfPresent(kernel_shape, "kernel_shape");
  if (kernel_shape.size() != input.tensor(0)->info.rank() - 2) {
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
  return std::unique_ptr<Op>(new AveragePoolOp(*this));
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> AveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new AveragePoolGradOp(this)));
  return upops;
}

AveragePoolGradOp::AveragePoolGradOp(AveragePoolOp *op_)
    : Op({"AveragePoolGrad", op_->pir, {}, getWillowDomain()}),
      unpooledInfo(op_->input.tensor(0)->info), cloneOfCreator(op_->clone()) {}

const std::vector<GradInOutMapper> &AveragePoolGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo =
      createAveragePoolGradInfo();
  return inInfo;
}

// The input to the average pool (PrePooled) is
// the input to the grad op at index 0.
int AveragePoolGradOp::getPrePooledIn() const { return 0; }

int AveragePoolGradOp::getPooledIn() const { return 1; }

int AveragePoolGradOp::getGradPooledIn() const { return 2; }

std::vector<GradInOutMapper>
AveragePoolGradOp::createAveragePoolGradInfo() const {
  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the average pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the average pool at index 0
  // etc for getPrePooledIn()
  return {{getGradPooledIn(), 0, GradOpInType::GRADOUT},
          {getPooledIn(), 0, GradOpInType::OUT},
          {getPrePooledIn(), 0, GradOpInType::IN}};
}

const std::map<int, int> &AveragePoolGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createAveragePoolGradOutToIn();
  return outInfo;
}

std::map<int, int> AveragePoolGradOp::createAveragePoolGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  return {{0, 0}};
}

void AveragePoolGradOp::setup() { output.tensor(0)->info = unpooledInfo; }

} // namespace willow
