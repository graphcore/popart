// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_SQUEEZECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SQUEEZECE_HPP

#include <popart/ces/identityce.hpp>

namespace popart {
class Op;

class ConstExprSqueeze : public ConstExprIdentity {
public:
  ConstExprSqueeze(Op *op_) : ConstExprIdentity(op_) {}
};

} // namespace popart

#endif
