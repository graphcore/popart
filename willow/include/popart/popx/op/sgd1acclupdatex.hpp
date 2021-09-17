// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1ACCLUPDATEX_HPP
#define GUARD_NEURALNET_SGD1ACCLUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class SGD1AcclUpdateOpx : public VarUpdateOpx {
public:
  SGD1AcclUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
