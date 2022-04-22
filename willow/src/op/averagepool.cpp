// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/op/averagepool.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/receptive.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

// TODO : Support "count_include_pad" T6249

AveragePoolOp::AveragePoolOp(
    const OperatorIdentifier &_opid,
    int64_t _countIncludePad,
    const std::vector<int64_t> &_kernelShape,
    const HasReceptiveFieldOp::ReceptiveOpAttributes &attributes,
    const Op::Settings &settings_)
    : HasReceptiveFieldOp(_opid, attributes, settings_),
      kernelShape(_kernelShape), countIncludePad(_countIncludePad) {}

void AveragePoolOp::setup0() const {}

Shape AveragePoolOp::getSpatialK() const {
  std::vector<int64_t> spatialK(getNSpatialDims());

  if (kernelShape.size() != inRank(getInIndex()) - 2) {
    throw error(
        "invalid kernel_shape, not same rank as the tensor operated on");
  }

  if (countIncludePad) {
    throw error("`count_include_pad` is not supported");
  }
  for (int spDim = 0; spDim < getNSpatialDims(); ++spDim) {
    spatialK[spDim] = kernelShape[spDim];
  }

  return spatialK;
}

std::unique_ptr<Op> AveragePoolOp::clone() const {
  return std::make_unique<AveragePoolOp>(*this);
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return getNInChans(); }

std::vector<std::unique_ptr<Op>> AveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AveragePoolGradOp>(*this));
  return upops;
}

void AveragePoolOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendOutlineAttributes(os);
  os.appendAttribute("kernel_shape", kernelShape);
  os.appendAttribute("count_include_pad", countIncludePad);
}

bool AveragePoolOp::canBeReplacedByIdentity() const {
  auto pads              = getPads();
  auto strides           = getStrides();
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
      creatorSpatialK(op_.getSpatialK()), creatorStrides(op_.getStrides()),
      creatorLowerPads(op_.getLowerPads()),
      creatorUpperPads(op_.getUpperPads()) {}

void AveragePoolGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("creatorSpatialK", creatorSpatialK);
  os.appendAttribute("creatorStrides", creatorStrides);
  os.appendAttribute("creatorLowerPads", creatorLowerPads);
  os.appendAttribute("creatorUpperPads", creatorUpperPads);
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
                                                {"ceil_mode", {"*"}},
                                                // Not currently supported
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
    [](const OpCreatorInfo &info) {
      HasReceptiveFieldOp::ReceptiveOpAttributes receptiveAttributes;
      receptiveAttributes.setFromAttributes(info.attributes);

      std::vector<int64_t> kernelShape =
          info.attributes.getAttribute<Attributes::Ints>("kernel_shape", {});
      int64_t countIncludePad =
          info.attributes.getAttribute<Attributes::Int>("count_include_pad", 0);

      return std::unique_ptr<Op>(new AveragePoolOp(info.opid,
                                                   countIncludePad,
                                                   kernelShape,
                                                   receptiveAttributes,
                                                   info.settings));
    },
    true);
} // namespace

} // namespace popart
