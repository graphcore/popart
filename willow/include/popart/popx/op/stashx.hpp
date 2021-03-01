// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STASHX_HPP
#define GUARD_NEURALNET_STASHX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

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

#endif
