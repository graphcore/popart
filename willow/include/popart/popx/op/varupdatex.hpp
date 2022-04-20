// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;

class VarUpdateOpx : public PopOpx {
public:
  VarUpdateOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}
};

} // namespace popx
} // namespace popart

#endif
