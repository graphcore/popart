// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEPRODX_HPP
#define GUARD_NEURALNET_REDUCEPRODX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class ReduceProdOpx : public PopOpx {
public:
  ReduceProdOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceProdGradOpx : public PopOpx {
public:
  ReduceProdGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
