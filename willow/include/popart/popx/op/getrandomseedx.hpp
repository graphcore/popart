// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GETRANDOMSEEDX_HPP
#define GUARD_NEURALNET_GETRANDOMSEEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class GetRandomSeedOpx : public PopOpx {
public:
  GetRandomSeedOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
