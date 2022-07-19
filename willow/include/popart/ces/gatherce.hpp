// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_GATHERCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_GATHERCE_HPP_

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprGather : public ConstExprOp {
public:
  ConstExprGather(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_GATHERCE_HPP_
