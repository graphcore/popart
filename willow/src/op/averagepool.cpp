// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <functional>
#include <memory>
#include <numeric>
#include <popart/error.hpp>
#include <popart/op/averagepool.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

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
  return std::make_unique<AveragePoolOp>(*this);
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return nInChans; }

std::vector<std::unique_ptr<Op>> AveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AveragePoolGradOp>(*this));
  return upops;
}

void AveragePoolOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendOutlineAttributes(os);
  os.appendAttribute("kernel_shape", kernelShape);
  os.appendAttribute("count_include_pad", countIncludePad);
  os.appendAttribute("ceil_mode", ceilMode);
  os.appendAttribute("auto_pad", getAutoPadStr(padType));
}

bool AveragePoolOp::canBeReplacedByIdentity() {
  int64_t padsSum        = std::accumulate(pads.begin(), pads.end(), 0);
  int64_t stridesProduct = std::accumulate(
      strides.begin(), strides.end(), 1, std::multiplies<int64_t>());
  int64_t kernelShapeProduct = std::accumulate(
      kernelShape.begin(), kernelShape.end(), 1, std::multiplies<int64_t>());
  if (padsSum == 0 && stridesProduct == 1 && kernelShapeProduct == 1) {
    return true;
  }
  return false;
}

AveragePoolGradOp::AveragePoolGradOp(const AveragePoolOp &op_)
    : Op(Onnx::GradOperators::AveragePoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(AveragePoolOp::getInIndex())),
      cloneOfCreator(op_.clone()) {}

void AveragePoolGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
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
       GradOpInType::GradOut},
      {getPooledInIndex(), AveragePoolOp::getOutIndex(), GradOpInType::Out},
      {getPrePooledInIndex(), AveragePoolOp::getInIndex(), GradOpInType::In}};
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
  return std::make_unique<AveragePoolGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    averagePoolOpDef({OpDefinition::Inputs({{"X", T}}),
                      OpDefinition::Outputs({{"Y", T}}),
                      OpDefinition::Attributes({// Deprecated
                                                {"auto_pad", {"NOTSET"}},
                                                // Not currently supported
                                                // {"ceil_mode", {"*"}},
                                                // {"count_include_pad", {"*"}},
                                                {"kernel_shape", {"*"}},
                                                {"pads", {"*"}},
                                                {"strides", {"*"}}})});

static OpCreator<AveragePoolOp> averagePoolOpCreator(
    OpDefinitions({
        {Onnx::Operators::AveragePool_1, averagePoolOpDef},
        {Onnx::Operators::AveragePool_7, averagePoolOpDef},
        {Onnx::Operators::AveragePool_10, averagePoolOpDef},
        {Onnx::Operators::AveragePool_11, averagePoolOpDef},
    }),
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

} // namespace popart
