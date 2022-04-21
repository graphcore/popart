// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/globalaveragepool.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

GlobalAveragePoolOp::GlobalAveragePoolOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void GlobalAveragePoolOp::setup() {
  // If the input is N x C x D1 x D2 then the output shape N x C x 1 x 1
  // If the input is N x C x D1 ... Dn then the output shape N x C x 1 x ... x 1
  auto gap = inShape(getInIndex());
  std::fill(gap.begin() + 2, gap.end(), 1);
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), gap};

  kernel =
      Shape(inShape(getInIndex()).begin() + 2, inShape(getInIndex()).end());
}

void GlobalAveragePoolOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("kernel", kernel);
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
  return std::make_unique<GlobalAveragePoolOp>(*this);
}

std::vector<std::unique_ptr<Op>> GlobalAveragePoolOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<GlobalAveragePoolGradOp>(*this));
  return upops;
}

GlobalAveragePoolGradOp::GlobalAveragePoolGradOp(const GlobalAveragePoolOp &op_)
    : Op(Onnx::GradOperators::GlobalAveragePoolGrad, op_.getSettings()),
      unpooledInfo(op_.inInfo(GlobalAveragePoolOp::getInIndex())),
      creatorSpatialK(op_.getSpatialK()), creatorStrides(op_.getStrides()),
      creatorLowerPads(op_.getLowerPads()),
      creatorUpperPads(op_.getUpperPads()) {}

void GlobalAveragePoolGradOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("creatorSpatialK", creatorSpatialK);
  os.appendAttribute("creatorStrides", creatorStrides);
  os.appendAttribute("creatorLowerPads", creatorLowerPads);
  os.appendAttribute("creatorUpperPads", creatorUpperPads);
}

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
       GradOpInType::GradOut},
      {getPooledInIndex(),
       GlobalAveragePoolOp::getOutIndex(),
       GradOpInType::Out},
      {getPrePooledInIndex(),
       GlobalAveragePoolOp::getInIndex(),
       GradOpInType::In}};
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

std::unique_ptr<Op> GlobalAveragePoolGradOp::clone() const {
  return std::make_unique<GlobalAveragePoolGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition globalAveragePoolOpDef({OpDefinition::Inputs({{"X", T}}),
                                            OpDefinition::Outputs({{"Y", T}}),
                                            OpDefinition::Attributes({})});

static OpCreator<GlobalAveragePoolOp> globalAveragePoolOpCreator(OpDefinitions(
    {{Onnx::Operators::GlobalAveragePool_1, globalAveragePoolOpDef}}));
} // namespace

} // namespace popart
