// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_IOTILECOPY_HPP
#define GUARD_NEURALNET_IOTILECOPY_HPP

#include <popart/op.hpp>

namespace popart {

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
                                   std::set<OpId> visited = {}) const final;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> visited = {}) const final;

  bool canShard() const override { return true; }
};
} // namespace popart

#endif
