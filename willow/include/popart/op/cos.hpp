// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COS_HPP
#define GUARD_NEURALNET_COS_HPP

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Ir;

class CosOp : public ElementWiseUnaryOp {
public:
  CosOp(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static OperatorIdentifier getOpId(const Ir &ir);
};

class CosGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  CosGradOp(const CosOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
