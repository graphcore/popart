// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_PADGRADX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_PADGRADX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_PADGRADX_HPP_
