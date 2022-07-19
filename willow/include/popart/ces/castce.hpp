// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_CASTCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_CASTCE_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_CES_CASTCE_HPP_
