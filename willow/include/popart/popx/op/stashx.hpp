// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_

#include <cstddef>
#include <popart/popx/popopx.hpp>

namespace snap {
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class StashOpx : public PopOpx {
public:
  StashOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  size_t hStashSize;
  bool canDynamicUpdateStash;

  void growDynamicStashUpdate(snap::program::Sequence &prog,
                              const snap::Tensor &stashIndex,
                              const snap::Tensor &inTensor,
                              const snap::Tensor &outTensor) const;
  void growStaticStashUpdate(snap::program::Sequence &prog,
                             const snap::Tensor &stashIndex,
                             const snap::Tensor &inTensor,
                             const snap::Tensor &outTensor) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_
