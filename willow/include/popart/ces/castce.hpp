// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprCast : public ConstExprOp {
public:
  ConstExprCast(Op *op);
  std::vector<char> compute() final;
};
} // namespace popart

#endif
