// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1VARUPDATEX_HPP
#define GUARD_NEURALNET_SGD1VARUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class SGD1VarUpdateOpx : public VarUpdateOpx {
public:
  SGD1VarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
