// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ISNAN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ISNAN_HPP_

#include <memory>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Ir;

class IsNaN : public ElementWiseUnaryBooleanOp {
public:
  IsNaN(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  static OperatorIdentifier getOpId(const Ir &ir);
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ISNAN_HPP_
