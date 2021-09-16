// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONEHOTX_HPP
#define GUARD_NEURALNET_ONEHOTX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class OnehotOpx : public PopOpx {
public:
  OnehotOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class OnehotGradOpx : public PopOpx {
public:
  OnehotGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
