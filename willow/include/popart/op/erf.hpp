// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ERF_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ERF_HPP_

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class ErfOp : public ElementWiseUnaryOp {
public:
  ErfOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ErfGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ErfGradOp(const ErfOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ERF_HPP_
