// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_SCALECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SCALECE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprScale : public ConstExprOp {
public:
  ConstExprScale(Op *op) : ConstExprOp(op) {}
  std::vector<char> compute() final;
};
} // namespace popart

#endif
