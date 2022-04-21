// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PLACEHOLDER_HPP
#define GUARD_NEURALNET_PLACEHOLDER_HPP

#include <popart/op.hpp>

namespace popart {

class PlaceholderOp : public Op {
public:
  PlaceholderOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
