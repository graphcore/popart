// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_REDUCEPROD_HPP
#define GUARD_NEURALNET_CONSTEXPRS_REDUCEPROD_HPP

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprReduceProd : public ConstExprOp {
public:
  ConstExprReduceProd(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
