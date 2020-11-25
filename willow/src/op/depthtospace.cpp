// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <numeric>
#include <popart/op/depthtospace.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::string toString(DepthToSpaceMode mode) {
  switch (mode) {
  case DepthToSpaceMode::DCR:
    return "DCR";
  case DepthToSpaceMode::CRD:
    return "CRD";
  default:
    throw error("Bad DepthToSpaceMode. Must be DCR or CRD.");
  }
}

std::ostream &operator<<(std::ostream &os, DepthToSpaceMode x) {
  os << toString(x);
  return os;
}

DepthSpaceBaseOp::DepthSpaceBaseOp(const OperatorIdentifier &_opid,
                                   int64_t blocksize_,
                                   const Op::Settings &settings_)
    : Op(_opid, settings_), blocksize(blocksize_) {}

std::unique_ptr<Op> DepthToSpaceOp::clone() const {
  return std::make_unique<DepthToSpaceOp>(*this);
}

DepthToSpaceOp::DepthToSpaceOp(const OperatorIdentifier &_opid,
                               int64_t blocksize_,
                               DepthToSpaceMode mode_,
                               const Op::Settings &settings_)
    : DepthSpaceBaseOp(_opid, blocksize_, settings_), mode(mode_) {}

void DepthSpaceBaseOp::setupHelper(const Shape &inShape) {

  std::string dName = debugName();
  if (blocksize == 0) {
    throw error("{}, blocksize is 0.", dName);
  }

  if (inShape.size() != 4) {
    throw error("Rank of input tensor of {} is {}."
                "But it must have 4 dimensions.",
                dName,
                inShape.size());
  }

  for (int i = 0; i < 4; i++) {
    if (inShape[i] == 0) {
      throw error("{} th component of {} input shape is 0.", i, dName);
    }
  }
}

void DepthToSpaceOp::setup() {
  // Check b, c, h, w = x.shape
  const auto inShape = inInfo(getInIndex()).shape();
  int64_t blocksize  = getBlocksize();
  setupHelper(inShape);

  if (inShape[1] % (blocksize * blocksize) != 0) {
    throw error("c component of DepthToSpaceOp input shape is not compatible"
                "with blocksize. b, c, h, w = input.shape. c must be divisible"
                "by blocksize^2. But c is {} and blocksize is {}.",
                inShape[1],
                blocksize);
  }

  if (!(mode == DepthToSpaceMode::DCR || mode == DepthToSpaceMode::CRD)) {
    throw error("Bad DepthToSpaceMode. Must be DCR or CRD.");
  }

  const Shape outShape = {inShape[0],
                          inShape[1] / (blocksize * blocksize),
                          inShape[2] * blocksize,
                          inShape[3] * blocksize};

  outInfo(getOutIndex()) = {inInfo(getInIndex()).data_type(), outShape};
}

void SpaceToDepthOp::setup() {
  // Check b, c, h, w = x.shape
  const auto inShape = inInfo(getInIndex()).shape();
  int64_t blocksize  = getBlocksize();
  setupHelper(inShape);

  if (inShape[2] % blocksize != 0) {
    throw error("h component of SpaceToDepthOp input shape is not compatible"
                "with blocksize. b, c, h, w = input.shape. h must be divisible"
                "by blocksize. But h is {} and blocksize is {}.",
                inShape[2],
                blocksize);
  }

  if (inShape[3] % blocksize != 0) {
    throw error("w component of SpaceToDepthOp input shape is not compatible"
                "with blocksize. b, c, h, w = input.shape. w must be divisible"
                "by blocksize. But w is {} and blocksize is {}.",
                inShape[3],
                blocksize);
  }

  const Shape outShape = {inShape[0],
                          inShape[1] * blocksize * blocksize,
                          inShape[2] / blocksize,
                          inShape[3] / blocksize};

  outInfo(getOutIndex()) = {inInfo(getInIndex()).data_type(), outShape};
}

std::vector<std::unique_ptr<Op>> DepthSpaceBaseOp::getGradOps() {
  throw error("No gradient operation for depth to space is available."
              "Depth to space should have been automatically replaced by "
              "Depth/Space to Space/Depth op pattern.");
}

SpaceToDepthOp::SpaceToDepthOp(const OperatorIdentifier &_opid,
                               int64_t blocksize_,
                               const Op::Settings &settings_)
    : DepthSpaceBaseOp(_opid, blocksize_, settings_) {}

std::unique_ptr<Op> SpaceToDepthOp::clone() const {
  return std::make_unique<SpaceToDepthOp>(*this);
}

namespace {

DepthToSpaceMode getDepthToSpaceModeFromString(const std::string &mode) {

  if (mode == "DCR") {
    return DepthToSpaceMode::DCR;
  } else if (mode == "CRD") {
    return DepthToSpaceMode::CRD;
  } else {
    throw error("Bad DepthToSpaceMode {}", mode);
  }
}

static OpDefinition::DataTypes T = {
    DataType::UINT8,
    DataType::UINT16,
    DataType::UINT32,
    DataType::UINT64,
    DataType::INT8,
    DataType::INT16,
    DataType::INT32,
    DataType::INT64,
    DataType::BFLOAT16,
    DataType::FLOAT16,
    DataType::FLOAT,
    DataType::DOUBLE,
    DataType::STRING,
    DataType::BOOL,
    DataType::COMPLEX64,
    DataType::COMPLEX128,
};

static OpDefinition
    depthtospace1_OpDef({OpDefinition::Inputs({{"input", T}}),
                         OpDefinition::Outputs({{"output", T}}),
                         OpDefinition::Attributes({{"blocksize", {"*"}}})});

static OpDefinition depthtospace11_OpDef(
    {OpDefinition::Inputs({{"input", T}}),
     OpDefinition::Outputs({{"output", T}}),
     OpDefinition::Attributes({{"blocksize", {"*"}}, {"mode", {"*"}}})});

static OpDefinition
    spacetodepthOpDef({OpDefinition::Inputs({{"input", T}}),
                       OpDefinition::Outputs({{"output", T}}),
                       OpDefinition::Attributes({{"blocksize", {"*"}}})});

static std::unique_ptr<Op> depthtospaceOpFactory(const OpCreatorInfo &info) {
  int64_t blocksize = static_cast<int64_t>(
      info.attributes.getAttribute<Attributes::Int>("blocksize", 0));

  DepthToSpaceMode mode = DepthToSpaceMode::DCR;

  if (info.opid != Onnx::Operators::DepthToSpace_1) {
    if (info.attributes.hasAttribute("mode")) {
      std::string modeString =
          info.attributes.getAttribute<Attributes::String>("mode");
      mode = getDepthToSpaceModeFromString(modeString);
    }
  }

  return std::make_unique<DepthToSpaceOp>(
      info.opid, blocksize, mode, info.settings);
}

static std::unique_ptr<Op> spacetodepthOpFactory(const OpCreatorInfo &info) {
  int64_t blocksize = static_cast<int64_t>(
      info.attributes.getAttribute<Attributes::Int>("blocksize", 0));

  return std::make_unique<SpaceToDepthOp>(info.opid, blocksize, info.settings);
}

static OpCreator<DepthToSpaceOp> depthtospaceOpCreator(
    {{Onnx::Operators::DepthToSpace_1, depthtospace1_OpDef},
     {Onnx::Operators::DepthToSpace_11, depthtospace11_OpDef},
     {Onnx::CustomOperators::DepthToSpace, depthtospace11_OpDef}},
    depthtospaceOpFactory,
    true);

static OpCreator<SpaceToDepthOp> spacetodepthOpCreator(
    {{Onnx::Operators::SpaceToDepth_1, spacetodepthOpDef}},
    spacetodepthOpFactory,
    true);

} // namespace

} // namespace popart
