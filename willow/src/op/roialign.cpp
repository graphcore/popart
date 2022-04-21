// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/op/roialign.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

RoiAlignOp::RoiAlignOp(const OperatorIdentifier &opid_,
                       const Op::Settings &settings_,
                       const float spatialScale_,
                       const uint64_t samplingRatio_,
                       const uint64_t alignedHeight_,
                       const uint64_t alignedWidth_)
    : Op(opid_, settings_), spatialScale(spatialScale_),
      samplingRatio(samplingRatio_), alignedHeight(alignedHeight_),
      alignedWidth(alignedWidth_) {}

std::unique_ptr<Op> RoiAlignOp::clone() const {
  return std::make_unique<RoiAlignOp>(*this);
}

void RoiAlignOp::setup() {
  // out shape and type info
  Shape bottomDataShape       = inInfo(0).shape();
  Shape bottomRoisShape       = inInfo(1).shape();
  Shape bottomBatchIndexShape = inInfo(2).shape();

  if (bottomDataShape.size() != 4) {
    throw error("The size of input data must be 4");
  }
  if (bottomRoisShape.size() != 2) {
    throw error("The size of input rois must be 2");
  }
  if (bottomBatchIndexShape.size() != 1) {
    throw error("The size of input batch_index must be 1");
  }

  Shape top_data_shape;
  top_data_shape.push_back(bottomRoisShape[0]);
  top_data_shape.push_back(bottomDataShape[1]);
  top_data_shape.push_back(alignedHeight);
  top_data_shape.push_back(alignedWidth);
  DataType dataType = inInfo(0).dataType();
  outInfo(0)        = {dataType, top_data_shape};
}

std::vector<std::unique_ptr<Op>> RoiAlignOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<RoiAlignGradOp>(*this));
  return upops;
}

void RoiAlignOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("spatial_scale", spatialScale);
  os.appendAttribute("sampling_ratio", samplingRatio);
  os.appendAttribute("output_height", alignedHeight);
  os.appendAttribute("output_width", alignedWidth);
}

RoiAlignGradOp::RoiAlignGradOp(const RoiAlignOp &op_)
    : Op(Onnx::GradOperators::RoiAlignGrad, op_.getSettings()),
      spatialScale(op_.getSpatialScale()),
      samplingRatio(op_.getSamplingRatio()),
      alignedHeight(op_.getAlignedHeight()),
      alignedWidth(op_.getAlignedWidth()) {}

std::unique_ptr<Op> RoiAlignGradOp::clone() const {
  return std::make_unique<RoiAlignGradOp>(*this);
}

void RoiAlignGradOp::setup() {
  // out shape and type info
  Shape topDiffShape          = inInfo(0).shape();
  Shape bottomRoisShape       = inInfo(1).shape();
  Shape bottomBatchIndexShape = inInfo(2).shape();
  Shape bottomDataShape       = inInfo(3).shape();

  if (bottomDataShape.size() != 4) {
    throw error("The size of input data must be 4");
  }
  if (bottomRoisShape.size() != 2) {
    throw error("The size of input rois must be 2");
  }
  if (bottomBatchIndexShape.size() != 1) {
    throw error("The size of input batch_index must be 1");
  }
  if (topDiffShape.size() != 4) {
    throw error("The size of input gradient must be 4");
  }

  Shape bottom_diff_shape = bottomDataShape;
  DataType dataType       = inInfo(0).dataType();
  outInfo(0)              = {dataType, bottom_diff_shape};
}

const std::vector<GradInOutMapper> &RoiAlignGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GradOut},
      {1, 1, GradOpInType::In},
      {2, 2, GradOpInType::In},
      {3, 0, GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &RoiAlignGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

void RoiAlignGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("spatial_scale", spatialScale);
  os.appendAttribute("sampling_ratio", samplingRatio);
  os.appendAttribute("output_height", alignedHeight);
  os.appendAttribute("output_width", alignedWidth);
}

namespace {
// register op
static OpDefinition::DataTypes roiAlignOpFLOATType = {DataType::FLOAT16,
                                                      DataType::FLOAT};
static OpDefinition::DataTypes batchIndexType      = {DataType::INT32};

static OpDefinition roiAlignOpDef(
    {OpDefinition::Inputs({{"bottom_data", roiAlignOpFLOATType},
                           {"bottom_rois", roiAlignOpFLOATType},
                           {"bottom_batch_index", batchIndexType}}),
     OpDefinition::Outputs({{"top_data", roiAlignOpFLOATType}}),
     OpDefinition::Attributes({{"spatial_scale", {"*"}},
                               {"sampling_ratio", {"*"}},
                               {"output_height", {"*"}},
                               {"output_width", {"*"}}})});

static OpCreator<RoiAlignOp> roiAlignOpCreator(
    OpDefinitions({{Onnx::Operators::RoiAlign_10, roiAlignOpDef}}),
    [](const OpCreatorInfo &info) {
      auto &attr     = info.attributes;
      auto &opid     = info.opid;
      auto &settings = info.settings;
      float spatialScale =
          attr.getAttribute<Attributes::Float>("spatial_scale", 1);
      uint64_t samplingRatio =
          attr.getAttribute<Attributes::Int>("sampling_ratio", 1);
      uint64_t alignedHeight =
          attr.getAttribute<Attributes::Int>("output_height", 7);
      uint64_t alignedWidth =
          attr.getAttribute<Attributes::Int>("output_width", 7);
      return std::unique_ptr<RoiAlignOp>(new RoiAlignOp(opid,
                                                        settings,
                                                        spatialScale,
                                                        samplingRatio,
                                                        alignedHeight,
                                                        alignedWidth));
    },
    true);
} // namespace
} // namespace popart