// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_IDENTITYCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_IDENTITYCE_HPP

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprIdentity : public ConstExprOp {
public:
  ConstExprIdentity(Op *);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
