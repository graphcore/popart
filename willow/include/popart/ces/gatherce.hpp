// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_GATHERCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_GATHERCE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprGather : public ConstExprOp {
public:
  ConstExprGather(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
