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
    throw error("Bad DepthToSpaceMode {}", static_cast<int>(mode));
  }
}

std::ostream &operator<<(std::ostream &os, DepthToSpaceMode x) {
  os << toString(x);
  return os;
}

DepthToSpaceBaseOp::DepthToSpaceBaseOp(const OperatorIdentifier &_opid,
                                       int64_t blocksize_,
                                       DepthToSpaceMode mode_,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_), blocksize(blocksize_), mode(mode_) {}

std::unique_ptr<Op> DepthToSpaceOp::clone() const {
  return std::make_unique<DepthToSpaceOp>(*this);
}

DepthToSpaceOp::DepthToSpaceOp(const OperatorIdentifier &_opid,
                               int64_t blocksize_,
                               DepthToSpaceMode mode_,
                               const Op::Settings &settings_)
    : DepthToSpaceBaseOp(_opid, blocksize_, mode_, settings_) {}

void DepthToSpaceBaseOp::setup() {

  if (!(mode == DepthToSpaceMode::DCR || mode == DepthToSpaceMode::CRD)) {
    throw error("Bad DepthToSpaceMode. Must be DCR or CRD.");
  }

  if (blocksize == 0) {
    throw error("DepthToSpaceOp, blocksize is 0.");
  }

  // Check b, c, h, w = x.shape
  const auto in_shape = inInfo(getInIndex()).shape();

  if (in_shape.size() != 4) {
    throw error("Rank of input tensor of depth to space op, "
                "DepthToSpaceOp, is {}. But it must have 4 dimensions.",
                in_shape.size());
  }

  for (int i = 0; i < 4; i++) {
    if (in_shape[i] == 0) {
      throw error("{} th component of DepthToSpaceOp input shape is 0.", i);
    }
  }

  if (in_shape[1] % (blocksize * blocksize) != 0) {
    throw error("c component of DepthToSpaceOp input shape is not compatible"
                "with blocksize. c must be divisible by blocksize^2."
                "But c is {} and blocksize is {}.",
                in_shape[1],
                blocksize);
  }

  const Shape out_shape = {in_shape[0],
                           in_shape[1] / (blocksize * blocksize),
                           in_shape[2] * blocksize,
                           in_shape[3] * blocksize};

  outInfo(getOutIndex()) = {inInfo(getInIndex()).data_type(), out_shape};
}

std::vector<std::unique_ptr<Op>> DepthToSpaceBaseOp::getGradOps() {
  throw error("No gradient operation for depth to space is available."
              "Depth to space should have been automatically replaced by "
              "DepthToSpaceOp pattern.");
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

static OpDefinition depthtospaceOpDef(
    {OpDefinition::Inputs({{"input", T}}),
     OpDefinition::Outputs({{"output", T}}),
     OpDefinition::Attributes({{"blocksize", {"*"}}, {"mode", {"*"}}})});

static std::unique_ptr<Op> depthtospaceOpFactory(const OpCreatorInfo &info) {
  int64_t blocksize = static_cast<int64_t>(
      info.attributes.getAttribute<Attributes::Int>("blocksize", 0));

  DepthToSpaceMode mode = DepthToSpaceMode::DCR;
  if (info.attributes.hasAttribute("mode")) {
    std::string modeString =
        info.attributes.getAttribute<Attributes::String>("mode");
    mode = getDepthToSpaceModeFromString(modeString);
  }

  return std::make_unique<DepthToSpaceOp>(
      info.opid, blocksize, mode, info.settings);
}

static OpCreator<DepthToSpaceOp> depthtospaceOpCreator(
    {{Onnx::Operators::DepthToSpace_11, depthtospaceOpDef}},
    depthtospaceOpFactory,
    true);
} // namespace

} // namespace popart
