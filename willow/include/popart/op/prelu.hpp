// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PRELU_HPP
#define GUARD_NEURALNET_PRELU_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class PReluOp : public ElementWiseBinaryOp {
public:
  PReluOp(const OperatorIdentifier &opid_, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
