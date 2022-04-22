// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_FLOORCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_FLOORCE_HPP

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprFloor : public ConstExprOp {
public:
  ConstExprFloor(Op *);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
