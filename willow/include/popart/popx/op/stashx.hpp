// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STASHX_HPP
#define GUARD_NEURALNET_STASHX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

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

#endif
