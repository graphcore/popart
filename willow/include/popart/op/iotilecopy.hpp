// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_OP_IOTILECOPY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_IOTILECOPY_HPP_

#include <memory>
#include <set>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
struct OperatorIdentifier;

class IoTileCopyOp : public Op {
public:
  IoTileCopyOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex,
                                   std::set<OpId> &visited) const final;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> &visited) const final;

  bool canShard() const override { return true; }
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_IOTILECOPY_HPP_
