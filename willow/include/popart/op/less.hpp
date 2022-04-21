// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LESS_HPP
#define GUARD_NEURALNET_LESS_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class LessOp : public BinaryComparisonOp {
public:
  LessOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
