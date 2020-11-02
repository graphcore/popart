// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATANH_HPP
#define GUARD_NEURALNET_ATANH_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class AtanhOp : public ElementWiseUnaryOp {
public:
  AtanhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
