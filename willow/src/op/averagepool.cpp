#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

// TODO : Support "count_include_pad" T6249

// TODO : Support ceilMode T9185

AveragePoolOp::AveragePoolOp(const OperatorIdentifier &_opid,
                             int64_t _countIncludePad,
                             int64_t _ceilMode,
                             const std::vector<int64_t> &_kernelShape,
                             const HasReceptiveFieldOp::Settings &settings_)
    : HasReceptiveFieldOp(_opid, settings_), kernelShape(_kernelShape),
      countIncludePad(_countIncludePad), ceilMode(_ceilMode) {

  // TODO : Use the count_include_pad for AveragePool-1
}

void AveragePoolOp::setup0() {}

void AveragePoolOp::setSpatialK() {
  spatialK.resize(nSpatialDims);

  if (kernelShape.size() != inRank(getInIndex()) - 2) {
    throw error(
        "invalid kernel_shape, not same rank as the tensor operated on");
  }

  if (countIncludePad) {
    throw error("`count_include_pad` is not supported");
  }
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatialK[spDim] = kernelShape[spDim];
  }
}

const AveragePoolOp *AveragePoolGradOp::getCloneOfCreator() const {
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
  upops.emplace_back(make_unique<AveragePoolGradOp>(*this));
  return upops;
}

void AveragePoolOp::appendAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendAttributes(os);
  os.appendAttribute("kernel_shape", kernelShape);
  os.appendAttribute("count_include_pad", countIncludePad);
  os.appendAttribute("ceil_mode", ceilMode);
}

AveragePoolGradOp::AveragePoolGradOp(const AveragePoolOp &op_)
    : Op(Onnx::GradOperators::AveragePoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(AveragePoolOp::getInIndex())),
      cloneOfCreator(op_.clone()) {}

void AveragePoolGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendForwardOp(getCloneOfCreator());
}

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

std::unique_ptr<Op> AveragePoolGradOp::clone() const {
  return make_unique<AveragePoolGradOp>(*this);
}

namespace {
static OpCreator<AveragePoolOp> averagePoolOpCreator(
    {Onnx::Operators::AveragePool_1,
     Onnx::Operators::AveragePool_7,
     Onnx::Operators::AveragePool_10},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      HasReceptiveFieldOp::Settings receptiveSettings(
          settings.graph, settings.name, settings.scope);
      receptiveSettings.setFromAttributes(attr);

      std::vector<int64_t> kernelShape =
          attr.getAttribute<Attributes::Ints>("kernel_shape", {});
      int64_t countIncludePad =
          attr.getAttribute<Attributes::Int>("count_include_pad", 0);
      int64_t ceilMode = attr.getAttribute<Attributes::Int>("ceil_mode", 0);

      return std::unique_ptr<Op>(new AveragePoolOp(
          _opid, countIncludePad, ceilMode, kernelShape, receptiveSettings));
    },
    true);
} // namespace

} // namespace poponnx
