// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_BUCKETIZEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_BUCKETIZEX_HPP_

#include "popart/popx/opx.hpp"
#include <popart/names.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class Bucketizex : public Opx {
public:
  Bucketizex(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override final;
  bool outputCreatedExternally(OutIndex index) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_BUCKETIZEX_HPP_
