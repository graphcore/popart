// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/op/maxpool.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/receptive.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace popart {

MaxPoolOp::MaxPoolOp(
    const OperatorIdentifier &_opid,
    const std::vector<int64_t> &kernelShape_,
    int64_t storageOrder_,
    const HasReceptiveFieldOp::ReceptiveOpAttributes &attributes,
    const Op::Settings &settings_)
    : HasReceptiveFieldOp(_opid, attributes, settings_),
      storageOrder(storageOrder_), kernelShape(kernelShape_) {}

void MaxPoolOp::setup0() const {

  if (storageOrder != 0) {
    throw error("storage_order != 0, not supported");
  }
}

Shape MaxPoolOp::getSpatialK() const {
  std::vector<int64_t> spatialK(getNSpatialDims());

  if (kernelShape.size() != inRank(getInIndex()) - 2) {
    throw error(
        "invalid kernel_shape, not same rank as the tensor operated on");
  }
  for (int spDim = 0; spDim < getNSpatialDims(); ++spDim) {
    spatialK[spDim] = kernelShape[spDim];
  }
  return spatialK;
}

std::unique_ptr<Op> MaxPoolOp::clone() const {
  return std::make_unique<MaxPoolOp>(*this);
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t MaxPoolOp::getNOutChans() const { return getNInChans(); }

std::vector<std::unique_ptr<Op>> MaxPoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<MaxPoolGradOp>(*this));
  return upops;
}

void MaxPoolOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendOutlineAttributes(os);
  os.appendAttribute("storage_order", storageOrder);
  os.appendAttribute("kernel_shape", kernelShape);
}

bool MaxPoolOp::canBeReplacedByIdentity() const {
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

MaxPoolGradOp::MaxPoolGradOp(const MaxPoolOp &op_)
    : Op(Onnx::GradOperators::MaxPoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(MaxPoolOp::getInIndex())),
      creatorSpatialK(op_.getSpatialK()), creatorStrides(op_.getStrides()),
      creatorLowerPads(op_.getLowerPads()),
      creatorUpperPads(op_.getUpperPads()) {}

void MaxPoolGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("creatorSpatialK", creatorSpatialK);
  os.appendAttribute("creatorStrides", creatorStrides);
  os.appendAttribute("creatorLowerPads", creatorLowerPads);
  os.appendAttribute("creatorUpperPads", creatorUpperPads);
}

const std::vector<GradInOutMapper> &MaxPoolGradOp::gradInputInfo() const {

  // the input to the grad-op at index getGradPooledIn()
  // is the gradient of the output of the max pool
  // at index 0.
  // the input to the grad-op at index getPooledIn()
  // is the output of the max pool at index 0
  // etc for getPrePooledIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradPooledInIndex(), MaxPoolOp::getOutIndex(), GradOpInType::GradOut},
      {getPooledInIndex(), MaxPoolOp::getOutIndex(), GradOpInType::Out},
      {getPrePooledInIndex(), MaxPoolOp::getInIndex(), GradOpInType::In}};
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

std::unique_ptr<Op> MaxPoolGradOp::clone() const {
  return std::make_unique<MaxPoolGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes I = {DataType::INT64};

static OpDefinition maxPoolOpV1Def({OpDefinition::Inputs({{"X", T}}),
                                    OpDefinition::Outputs({{"Y", T}}),
                                    OpDefinition::Attributes({
                                        {"auto_pad", {"NOTSET"}},
                                        {"kernel_shape", {"*"}},
                                        {"pads", {"*"}},
                                        {"strides", {"*"}},
                                    })});

static OpDefinition maxPoolOpV8Def({OpDefinition::Inputs({{"X", T}}),
                                    OpDefinition::Outputs({
                                        {"Y", T},
                                        //{"Indices", I } Not supported?
                                    }),
                                    OpDefinition::Attributes({
                                        {"auto_pad", {"NOTSET"}},
                                        {"kernel_shape", {"*"}},
                                        {"pads", {"*"}},
                                        {"storage_order", {"1"}},
                                        {"strides", {"*"}},
                                    })});

static OpDefinition maxPoolOpDef({OpDefinition::Inputs({{"X", T}}),
                                  OpDefinition::Outputs({
                                      {"Y", T},
                                      //{"Indices", I } Not supported?
                                  }),
                                  OpDefinition::Attributes({
                                      {"auto_pad", {"NOTSET"}},
                                      {"ceil_mode", {"*"}},
                                      {"dilations", {"*"}},
                                      {"kernel_shape", {"*"}},
                                      {"pads", {"*"}},
                                      {"storage_order", {"1"}},
                                      {"strides", {"*"}},
                                  })});

static OpCreator<MaxPoolOp> maxPoolOpCreator(
    OpDefinitions({
        {Onnx::Operators::MaxPool_1, maxPoolOpV1Def},
        {Onnx::Operators::MaxPool_8, maxPoolOpV8Def},
        {Onnx::Operators::MaxPool_10, maxPoolOpDef},
        {Onnx::Operators::MaxPool_11, maxPoolOpDef},
    }),
    [](const OpCreatorInfo &info) {
      HasReceptiveFieldOp::ReceptiveOpAttributes receptiveAttributes;
      receptiveAttributes.setFromAttributes(info.attributes);

      int64_t storageOrder =
          info.attributes.getAttribute<Attributes::Int>("storage_order", 0);
      std::vector<int64_t> kernelShape =
          info.attributes.getAttribute<Attributes::Ints>("kernel_shape", {});

      if (info.opid.version >= 10) {
        // opset 10 introduced the dilations parameter, but poplibs pooling does
        // not appear to support this.
        std::vector<int64_t> dilations =
            info.attributes.getAttribute<Attributes::Ints>("dilations", {});
        for (auto d : dilations) {
          if (d != 1) {
            throw error("MaxPool has dilations {}. Dilations of value other "
                        "than 1 are currently not supported.",
                        dilations);
          }
        }
      }

      return std::unique_ptr<Op>(new MaxPoolOp(info.opid,
                                               kernelShape,
                                               storageOrder,
                                               receptiveAttributes,
                                               info.settings));
    },
    true);
} // namespace

} // namespace popart
