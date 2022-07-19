// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_FLATTENCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_FLATTENCE_HPP_

#include <popart/ces/identityce.hpp>

namespace popart {
class Op;

class ConstExprFlatten : public ConstExprIdentity {
public:
  ConstExprFlatten(Op *_op_) : ConstExprIdentity(_op_) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_FLATTENCE_HPP_
