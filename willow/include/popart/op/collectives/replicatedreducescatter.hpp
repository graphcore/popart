// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDREDUCESCATTER_HPP
#define GUARD_NEURALNET_REPLICATEDREDUCESCATTER_HPP

#include <popart/op/collectives/collectives.hpp>

namespace popart {

class ReplicatedReduceScatterOp : public CollectivesBaseOp {
public:
  ReplicatedReduceScatterOp(const OperatorIdentifier &_opid,
                            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

} // namespace popart

#endif
