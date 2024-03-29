// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ARGEXTREMA_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ARGEXTREMA_HPP_

#include <cstdint>
#include <memory>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// The base class for an op that wants to choose some extreme values from an
// input tensor.
class ArgExtremaOp : public Op {
public:
  ArgExtremaOp(const OperatorIdentifier &_opid,
               int64_t axis,
               int64_t keepdims,
               const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  int64_t getKeepDims() const;
  int64_t getAxis() const;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

private:
  void validateAxis() const;

  const int64_t keepdims;
  const int64_t axis;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ARGEXTREMA_HPP_
