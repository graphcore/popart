// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TILE_HPP
#define GUARD_NEURALNET_TILE_HPP

#include <popart/op.hpp>

namespace popart {

// This Op is based on the ONNX Operator described at
// github.com/onnx/onnx/blob/master/docs/Operators.md#tile
// but it is slightly different: this Op is static w.r.t. the
// 'Repeats' input
class TileOp : public Op {
public:
  TileOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  // This will be used by TileGradOp
  TileOp(const OperatorIdentifier &_opid,
         const std::vector<int64_t> &repeats_,
         const Shape &outShape_,
         const Op::Settings &settings_);

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> clone() const override;
  void setup() final;
  virtual void connectInTensor(InIndex, TensorId) final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  const Shape &getOutShape();

  const std::vector<int64_t> &getRepeats() const;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // The shape of the data output tensor
  Shape outShape;

  // The number of repeated copies across the input dimensions
  std::vector<int64_t> repeats;
};

// The gradient of tile is the reverse of the
// tile (which is a slice)
class TileGradOp : public TileOp {
public:
  TileGradOp(const TileOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif
