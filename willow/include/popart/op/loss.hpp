// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LOSS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LOSS_HPP_

#include <memory>
#include <string>
#include <popart/op.hpp>

namespace popart {
struct OperatorIdentifier;

class LossOp : public Op {
public:
  LossOp(const OperatorIdentifier &_opid,
         const Op::Settings &settings_,
         const ReductionType reduction_);
  std::unique_ptr<Op> clone() const override = 0;

  bool isLossOp() const override;

  static std::string reductionTypeToString(ReductionType reduction);
  static ReductionType reductionTypeFromString(std::string reduction);

  ReductionType getReductionType() const { return reduction_type_; }

private:
  const ReductionType reduction_type_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_LOSS_HPP_
