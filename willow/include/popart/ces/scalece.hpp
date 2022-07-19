// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_SCALECE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_SCALECE_HPP_

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprScale : public ConstExprOp {
public:
  ConstExprScale(Op *op) : ConstExprOp(op) {}
  std::vector<char> compute() final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_SCALECE_HPP_
