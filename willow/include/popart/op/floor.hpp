// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FLOOR_HPP
#define GUARD_NEURALNET_FLOOR_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class FloorOp : public ElementWiseUnaryOp {
public:
  FloorOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class FloorInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  FloorInplaceOp(const FloorOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
