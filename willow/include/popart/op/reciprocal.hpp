// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RECIPROCAL_HPP
#define GUARD_NEURALNET_RECIPROCAL_HPP

#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {
struct OperatorIdentifier;

class ReciprocalOp : public ElementWiseUnaryOp {
public:
  ReciprocalOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReciprocalGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ReciprocalGradOp(const ReciprocalOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
