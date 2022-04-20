// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PADGRADX_HPP
#define GUARD_NEURALNET_PADGRADX_HPP

#include <popart/popx/op/slicex.hpp>

namespace popart {
class Op;

namespace popx {
class Devicex;

class PadGradOpx : public SliceOpx {
public:
  PadGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
