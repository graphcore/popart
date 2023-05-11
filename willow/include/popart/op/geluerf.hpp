// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_GELUERF_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_GELUERF_HPP_

#include <memory>
#include <tuple>
#include <vector>

#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

class GeluErfOp : public ElementWiseUnaryOp {
public:
  GeluErfOp(const OperatorIdentifier &opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class GeluErfInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  GeluErfInplaceOp(const GeluErfOp &);
  GeluErfInplaceOp(const Op::Settings &opSettings);

  std::unique_ptr<Op> clone() const final;
};

class GeluErfGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  GeluErfGradOp(const GeluErfOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_GELUERF_HPP_
