// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ISINF_HPP
#define GUARD_NEURALNET_ISINF_HPP

#include <memory>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Ir;

class IsInf : public ElementWiseUnaryBooleanOp {
public:
  IsInf(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  static OperatorIdentifier getOpId(const Ir &ir);
};

} // namespace popart

#endif
