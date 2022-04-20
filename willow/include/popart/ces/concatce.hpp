// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP

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

#endif
