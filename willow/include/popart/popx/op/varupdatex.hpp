// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_VARUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_VARUPDATEX_HPP_

#include <popart/popx/opx.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;

class VarUpdateOpx : public Opx {
public:
  VarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_VARUPDATEX_HPP_
