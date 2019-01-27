#include <poponnx/error.hpp>
#include <poponnx/op/maxpool.hpp>

#include <poponnx/makeunique.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

MaxPoolOp::MaxPoolOp(const OperatorIdentifier &_opid,
                     const std::vector<int64_t> &kernelShape_,
                     int64_t storageOrder_,
                     const HasReceptiveFieldOp::Settings &settings_)
    : HasReceptiveFieldOp(_opid, settings_), storageOrder(storageOrder_),
      kernelShape(kernelShape_) {}

void MaxPoolOp::setup0() {

  if (storageOrder != 0) {
    throw error("storage_order != 0, not supported");
  }
}

void MaxPoolOp::setSpatialK() {
  spatialK.resize(nSpatialDims);

  if (kernelShape.size() != inRank(getInIndex()) - 2) {
    throw error(
        "invalid kernel_shape, not same rank as the tensor operated on");
  }
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatialK[spDim] = kernelShape[spDim];
  }
}

const MaxPoolOp *MaxPoolGradOp::getCloneOfCreator() {
  return dynamic_cast<MaxPoolOp *>(cloneOfCreator.get());
}

std::unique_ptr<Op> MaxPoolOp::clone() const {
  return make_unique<MaxPoolOp>(*this);
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t MaxPoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> MaxPoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<MaxPoolGradOp>(*this));
  return upops;
}

void MaxPoolOp::appendAttributes(std::stringstream &ss,
                                 const std::string &tab) const {
  HasReceptiveFieldOp::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "storage_order", storageOrder);
  appendAttribute(ss, tab, "kernel_shape", kernelShape);
}

MaxPoolGradOp::MaxPoolGradOp(const MaxPoolOp &op_)
    : Op(Onnx::GradOperators::MaxPoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(MaxPoolOp::getInIndex())),
      cloneOfCreator(op_.clone()) {}

const std::vector<GradInOutMapper> &MaxPoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the max pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the max pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledInIndex(), MaxPoolOp::getOutIndex(), GradOpInType::GRADOUT},
      {getPooledInIndex(), MaxPoolOp::getOutIndex(), GradOpInType::OUT},
      {getPrePooledInIndex(), MaxPoolOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

// The input to the max pool (PrePooled) is
// the input to the grad op at index 0.

const std::map<int, int> &MaxPoolGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MaxPoolOp::getInIndex()}};
  return outInfo;
}

void MaxPoolGradOp::setup() { outInfo(getOutIndex()) = unpooledInfo; }

namespace {
static OpCreator<MaxPoolOp> maxPoolOpxCreator(
    {Onnx::Operators::MaxPool_8, Onnx::Operators::MaxPool_1},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      HasReceptiveFieldOp::Settings receptiveSettings(settings.ir,
                                                      settings.name);
      receptiveSettings.setFromAttributes(attr);

      int64_t storageOrder =
          attr.getAttribute<Attributes::Int>("storage_order", 0);
      std::vector<int64_t> kernelShape =
          attr.getAttribute<Attributes::Ints>("kernel_shape", {});

      return std::unique_ptr<Op>(
          new MaxPoolOp(_opid, kernelShape, storageOrder, receptiveSettings));
    },
    true);
} // namespace

} // namespace poponnx
