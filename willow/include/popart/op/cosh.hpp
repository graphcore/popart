// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COSH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COSH_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COSH_HPP_
