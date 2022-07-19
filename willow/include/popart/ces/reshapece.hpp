// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_RESHAPECE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_RESHAPECE_HPP_

#include <popart/ces/identityce.hpp>

namespace popart {
class Op;

class ConstExprReshape : public ConstExprIdentity {
public:
  ConstExprReshape(Op *_op_) : ConstExprIdentity(_op_) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_RESHAPECE_HPP_
