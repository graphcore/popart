// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORREMAP_HPP
#define GUARD_NEURALNET_TENSORREMAP_HPP

#include <popart/op.hpp>

namespace popart {

/**
 * Enum describing how the tensor layout should be remapped during the forward
 * and backward pass (backward pass remapping requires the Op to exist in the IR
 * before autodiff).
 */
enum class TensorRemapType {
  /// Remap the tensor in the forward pass, reverse-apply the remapping in the
  /// backward pass
  FwdBwdReverse = 0,
  /// Remap the tensor in the forward pass and backward pass independently
  FwdBwd,
  /// Only remap the tensor in the forward pass, use identity
  /// for the backward pass
  Fwd
};

/**
 * Op that creates a new output tensor with tensor layout created by downstream
 * consumers, and then copies the input tensor to the output tensor.
 * Can improve tile memory liveness if the tensor without remapping is
 * unsuitable for downstream consumers.
 * Should only be used if actual issues occur, since remapping clones the tensor
 * and can introduce more rearrangement and data copies than necessary.
 */
class TensorRemapOp : public Op {
public:
  TensorRemapOp(const OperatorIdentifier &,
                const TensorRemapType &,
                const Op::Settings &);
  TensorRemapOp(const TensorRemapOp &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static InIndex getRefInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  TensorRemapType getTensorRemapType() const { return remap_type; }

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool isOutlineable() const final { return true; };

private:
  TensorRemapType remap_type;
};

} // namespace popart

#endif
