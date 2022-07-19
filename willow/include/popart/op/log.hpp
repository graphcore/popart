// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LOG_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LOG_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_LOG_HPP_
