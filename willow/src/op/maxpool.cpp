#include <poponnx/error.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

MaxPoolOp::MaxPoolOp(const onnx::NodeProto &node, Ir *_pir)
    : HasReceptiveFieldOp(node, _pir) {}

void MaxPoolOp::setup0() {
  int64_t storage_order = 0;
  nAtts.setIfPresent(storage_order, "storage_order");
  if (storage_order != 0) {
    throw error("storage_order != 0, not supported");
  }
}

void MaxPoolOp::setSpatialK() {
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

const MaxPoolOp *MaxPoolGradOp::getCloneOfCreator() {
  return dynamic_cast<MaxPoolOp *>(cloneOfCreator.get());
}

std::unique_ptr<Op> MaxPoolOp::clone() const {
  return std::unique_ptr<Op>(new MaxPoolOp(*this));
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t MaxPoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> MaxPoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new MaxPoolGradOp(this)));
  return upops;
}

MaxPoolGradOp::MaxPoolGradOp(MaxPoolOp *op_)
    : Op({"MaxPoolGrad", op_->pir, {}, getPoponnxDomain()}),
      unpooledInfo(op_->input.tensor(0)->info), cloneOfCreator(op_->clone()) {}

const std::vector<GradInOutMapper> &MaxPoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the max pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the max pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledIn(), 0, GradOpInType::GRADOUT},
      {getPooledIn(), 0, GradOpInType::OUT},
      {getPrePooledIn(), 0, GradOpInType::IN}};
  return inInfo;
}

// The input to the max pool (PrePooled) is
// the input to the grad op at index 0.
int MaxPoolGradOp::getPrePooledIn() const { return 0; }

int MaxPoolGradOp::getPooledIn() const { return 1; }

int MaxPoolGradOp::getGradPooledIn() const { return 2; }

const std::map<int, int> &MaxPoolGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void MaxPoolGradOp::setup() { output.tensor(0)->info = unpooledInfo; }

} // namespace willow
