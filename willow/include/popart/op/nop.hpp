// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NOP_HPP
#define GUARD_NEURALNET_NOP_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

// The NopOp is functionally the same as the IdentityOp.
// The difference is that it will not be removed by patterns.
// It is meant as a debug operation to prevent patterns happening or prevent
// inplacing.
class NopOp : public ElementWiseUnaryOp {
public:
  NopOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
