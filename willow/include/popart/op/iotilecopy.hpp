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

  VGraphIdAndIoTile getIntrospectionInVirtualGraphId(InIndex) const final;
  VGraphIdAndIoTile getIntrospectionOutVirtualGraphId(OutIndex) const final;
};
} // namespace popart

#endif
