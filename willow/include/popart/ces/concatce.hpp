// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_CONCATCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_CONCATCE_HPP_

#include <cstdint>
#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprConcat : public ConstExprOp {
public:
  ConstExprConcat(Op *);
  std::vector<char> compute() final;

private:
  int64_t input_count;
  int64_t axis;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_CONCATCE_HPP_
