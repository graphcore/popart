// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_UNSQUEEZECE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_UNSQUEEZECE_HPP_

#include <popart/ces/identityce.hpp>

namespace popart {
class Op;

class ConstExprUnsqueeze : public ConstExprIdentity {
public:
  ConstExprUnsqueeze(Op *op_) : ConstExprIdentity(op_) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_UNSQUEEZECE_HPP_
