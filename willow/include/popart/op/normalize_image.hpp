// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGE_HPP_

#include <memory>
#include <string>

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordebuginfo.hpp>

namespace popart {

class Ir;

// Normalize image op is the optimized version of image (in (B,H,W,C) format)
// preprocessing op to normalize image input in an image application. See
// `NormaliseImage
// <https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs/popops/NormaliseImage.html>`_
// for details.
//
// IPU architecture is well-suited to efficiently handle
// convolutions over four-channel tensors, however it is common for images to be
// represented with three channels. In order to obtain better IPU performance,
// both from a latency and memory standpoint, this op will pad three channels
// input to four channels automatically on-device after the data has been
// transferred to the IPU.
//
// The output is a padded and normalized image which always has four channels in
// the last dimension.
class NormalizeImageOp : public popart::Op {
public:
  NormalizeImageOp(const popart::OperatorIdentifier &_opid,
                   const popart::Op::Settings &settings_,
                   float _scale);

  std::unique_ptr<Op> clone() const override;
  static OperatorIdentifier getOpId(const Ir &ir);

  void setup() override;

  static popart::InIndex getImageInIndex() { return 0; }

  static popart::InIndex getOffsetsInIndex() { return 1; }

  static popart::InIndex getScalesInIndex() { return 2; }

  static popart::OutIndex getOutIndex() { return 0; }

  static std::string opName() { return "NormalizeImageOp"; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return false; };

  const popart::Tensor *imgIn() const;

  const popart::Tensor *offsetsIn() const;

  const popart::Tensor *scalesIn() const;

  float getScale() const { return scale; }

  bool canBeReplacedByIdentity() const override;

  bool verifyInputShapes(popart::Shape imgShape);

  // zero padded the image from 3 channels shape to 4 channels shape
  popart::Shape paddedShape(popart::Shape imgShape,
                            popart::Shape offsetsShape,
                            popart::Shape scalesShape);

private:
  float scale;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGE_HPP_
