// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <iosfwd>
#include <limits>
#include <vector>

#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/error.hpp>
#include <popart/graphcoreoperators.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/normalize_image.hpp>
#include <popart/opmanager.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

NormalizeImageOp::NormalizeImageOp(const popart::OperatorIdentifier &_opid,
                                   const popart::Op::Settings &settings_,
                                   float _scale)
    : Op(_opid, settings_), scale(_scale) {}

std::unique_ptr<popart::Op> NormalizeImageOp::clone() const {
  return std::make_unique<NormalizeImageOp>(*this);
}

OperatorIdentifier NormalizeImageOp::getOpId(const Ir &ir) {
  return Onnx::CustomOperators::NormalizeImageOpId;
}

void NormalizeImageOp::setup() {
  auto iDtype = imgIn()->info.dataType();
  auto oDtype =
      iDtype == popart::DataType::UINT8 ? popart::DataType::FLOAT16 : iDtype;

  // Define the shape of the output tensor
  outInfo(getOutIndex()) = {oDtype,
                            paddedShape(imgIn()->info.shape(),
                                        offsetsIn()->info.shape(),
                                        scalesIn()->info.shape())};
}

const popart::Tensor *NormalizeImageOp::imgIn() const {
  return inTensor(getImageInIndex());
}

const popart::Tensor *NormalizeImageOp::offsetsIn() const {
  return inTensor(getOffsetsInIndex());
}

const popart::Tensor *NormalizeImageOp::scalesIn() const {
  return inTensor(getScalesInIndex());
}

bool NormalizeImageOp::canBeReplacedByIdentity() const {
  if (!hasInput(getImageInIndex())) {
    const popart::Tensor *img = imgIn();
    auto imgShape             = img->info.shape();
    if (imgShape[imgShape.size() - 1] == 4) {
      return true;
    }
  }
  return false;
}

bool NormalizeImageOp::verifyInputShapes(popart::Shape imgShape) {
  if (imgShape.size() < 2 || imgShape[imgShape.size() - 1] != 3) {
    throw popart::error("requires the inner dimension to be 3.");
  }
  return true;
}

popart::Shape NormalizeImageOp::paddedShape(popart::Shape imgShape,
                                            popart::Shape offsetsShape,
                                            popart::Shape scalesShape) {
  verifyInputShapes(imgShape);
  popart::Shape ret   = imgShape;
  ret[ret.size() - 1] = 4;
  return ret;
}

// instantiate popart Op
namespace {

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition
    normalizeImageOpDef({popart::OpDefinition::Inputs(
                             {{"image", T}, {"offsets", T}, {"scales", T}}),
                         popart::OpDefinition::Outputs({{"output", T}}),
                         popart::OpDefinition::Attributes({{"scale", {"*"}}})});

static popart::OpCreator<NormalizeImageOp> normalizeImageOpCreator(
    popart::OpDefinitions({
        {Onnx::CustomOperators::NormalizeImageOpId, normalizeImageOpDef},
    }),
    [](const popart::OpCreatorInfo &info) {
      float scale =
          info.attributes.getAttribute<popart::Attributes::Float>("scale", 1.0);
      if (std::abs(scale) < std::numeric_limits<float>::epsilon()) {
        throw popart::error(
            "[normalizeImageOpCreator] scale should be greater than 0.");
      }
      return std::make_unique<NormalizeImageOp>(
          info.opid, info.settings, scale);
    },
    true);

} // namespace

} // namespace popart
