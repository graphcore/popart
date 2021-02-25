// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <map>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

enum class ScaleByReplication { Yes, No };

class LossOp : public Op {
public:
  LossOp(const OperatorIdentifier &_opid,
         const Op::Settings &settings_,
         const ReductionType reduction_);

  bool isLossOp() const override;

  static std::string reductionTypeToString(ReductionType reduction);
  static ReductionType reductionTypeFromString(std::string reduction);

  // TODO : remove as part of T34809
  ScaleByReplication getScaleByReplication(ReductionType reduction) const;

  ReductionType getReductionType() const { return reduction_type_; }

private:
  const ReductionType reduction_type_;
};

} // namespace popart

#endif
