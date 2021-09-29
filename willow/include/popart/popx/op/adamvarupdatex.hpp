// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMVARUPDATEX_HPP
#define GUARD_NEURALNET_ADAMVARUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class AdamVarUpdateOpx : public VarUpdateOpx {
public:
  AdamVarUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // does not create inputs
};

} // namespace popx
} // namespace popart

#endif
