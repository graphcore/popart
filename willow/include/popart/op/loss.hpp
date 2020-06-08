// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <map>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

// When weight updates of a batch are computed in one go, we
// are reducing over the gradients of the whole minibatch.
// What type of reduction should this be?
// Sum : Sum the output of the loss values and do not scale the gradient
// Mean : Take the mean of the loss values and divide the gradient by the number
//        of samples
// NoReduction : Leave the loss values as they are and do not scale the gradient
enum class ReductionType { Sum = 0, Mean, NoReduction };

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
