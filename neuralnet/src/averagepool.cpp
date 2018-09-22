#include <neuralnet/averagepool.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

AveragePoolOp::AveragePoolOp(const onnx::NodeProto &node, Graph *pgraph)
    : HasReceptiveFieldOp(node, pgraph) {}

void AveragePoolOp::setup0() {}

void AveragePoolOp::setSpatial() {
  spatial.reserve(nSpatialDims);
  std::vector<int64_t> kernel_shape;
  nAtts.setIfPresent(kernel_shape, "kernel_shape");
  if (kernel_shape.size() != input.tensor(0)->info.rank() - 2) {
    throw error("invald kernel_shape, not same rank as tensor operate on");
  }
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatial.push_back(kernel_shape[spDim]);
  }
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> AveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(
  std::unique_ptr<Op>(new AveragePoolGradOp(this)));
  return upops;
}

AveragePoolGradOp::AveragePoolGradOp(AveragePoolOp *op_)
    : Op({"AveragePoolGrad", op_->pgraph, {}, getNeuralNetDomain()}),
      averagePoolOp(op_) {

  std::cout << "AveragePoolGradOp constructed" << std::endl;
}

bool AveragePoolOp::readyToCreateGradients(std::set<int> & s0) const{
  return s0.size() == output.n();
}


const std::vector<GradInOutMapper> & AveragePoolGradOp::gradInputInfo() const{
  static const std::vector<GradInOutMapper> inInfo = createGradInputInfo();
  return inInfo;
}

std::vector<GradInOutMapper> AveragePoolGradOp::createGradInputInfo()  const{
  // the input to the grad-op at index 0 is the gradient 
  // of the output of the non-grad-op at index 0.
  return {{0, 0, GradOpInType::GRADOUT}};
}

const std::map<int, int> &AveragePoolGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createGradOutToNonGradInInfo();
  return outInfo;
}

Op * AveragePoolGradOp::getNonGradOp() {
  return averagePoolOp;
}


int AveragePoolGradOp::getNonGradInIndex(int gradOpOutIndex) const{
  return gradOutToNonGradIn().at(gradOpOutIndex);
}

std::map<int, int> AveragePoolGradOp::createGradOutToNonGradInInfo() const{
  // the grad-op output at index 0 corresponds 
  // to the non-grad-op's input at index 0
  return {{0,0}};
}


const std::map<int, Tensor *> & AveragePoolGradOp::gradOutMap(){
  return output.tensorMap();
}

} // namespace neuralnet
