// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOG_HPP
#define GUARD_NEURALNET_LOG_HPP

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class LogOp : public ElementWiseUnaryOp {
public:
  LogOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class LogGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  LogGradOp(const LogOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
