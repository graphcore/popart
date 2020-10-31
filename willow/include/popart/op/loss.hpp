// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <map>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

class LossOp : public Op {
public:
  LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  LossOp(const Op &);

  bool isLossOp() const override;

  static std::string reductionTypeToString(ReductionType reduction);
  static ReductionType reductionTypeFromString(std::string reduction);
};

} // namespace popart

#endif
