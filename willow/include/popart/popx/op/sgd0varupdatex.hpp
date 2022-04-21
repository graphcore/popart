// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD0VARUPDATEX_HPP
#define GUARD_NEURALNET_SGD0VARUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class SGD0VarUpdateOpx : public VarUpdateOpx {
public:
  SGD0VarUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
