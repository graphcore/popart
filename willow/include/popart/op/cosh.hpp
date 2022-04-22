// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COSH_HPP
#define GUARD_NEURALNET_COSH_HPP

#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

// TODO T8611 : make UnaryOp
class CoshOp : public Op {
public:
  CoshOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
