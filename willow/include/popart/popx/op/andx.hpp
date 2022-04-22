// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ANDX_HPP
#define GUARD_NEURALNET_ANDX_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

class Op;

namespace popx {
class Devicex;

class AndOpx : public BinaryComparisonOpx {
public:
  AndOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
