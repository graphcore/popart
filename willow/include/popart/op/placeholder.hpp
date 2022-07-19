// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_PLACEHOLDER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_PLACEHOLDER_HPP_

#include <memory>
#include <popart/op.hpp>

namespace popart {
struct OperatorIdentifier;

class PlaceholderOp : public Op {
public:
  PlaceholderOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_PLACEHOLDER_HPP_
