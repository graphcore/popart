// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ABSX_HPP
#define GUARD_NEURALNET_ABSX_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class AbsOpx : public ElementWiseUnaryOpx {
public:
  AbsOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
