// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WHERE_HPP
#define GUARD_NEURALNET_WHERE_HPP

#include <popart/op.hpp>

namespace popart {

class WhereOp : public Op {
public:
  WhereOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  // Inputs
  static InIndex conditionInIndex() { return 0; }
  static InIndex xInIndex() { return 1; }
  static InIndex yInIndex() { return 2; }

  // Ouputs
  static OutIndex outIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
