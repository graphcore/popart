// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_

#include <cstddef>
#include <popart/popx/opx.hpp>

namespace poplar {
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class StashOpx : public Opx {
public:
  StashOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  size_t hStashSize;
  bool canDynamicUpdateStash;

  void growDynamicStashUpdate(poplar::program::Sequence &prog,
                              const poplar::Tensor &stashIndex,
                              const poplar::Tensor &inTensor,
                              const poplar::Tensor &outTensor) const;
  void growStaticStashUpdate(poplar::program::Sequence &prog,
                             const poplar::Tensor &stashIndex,
                             const poplar::Tensor &inTensor,
                             const poplar::Tensor &outTensor) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_STASHX_HPP_
