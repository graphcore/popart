// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_FLATTENCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_FLATTENCE_HPP

#include <popart/ces/identityce.hpp>

namespace popart {

class ConstExprFlatten : public ConstExprIdentity {
public:
  ConstExprFlatten(Op *_op_) : ConstExprIdentity(_op_) {}
};

} // namespace popart

#endif
