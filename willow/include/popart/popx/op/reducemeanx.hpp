// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEMEANX_HPP
#define GUARD_NEURALNET_REDUCEMEANX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceMeanOpx : public PopOpx {
public:
  ReduceMeanOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceMeanGradOpx : public PopOpx {
public:
  ReduceMeanGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
