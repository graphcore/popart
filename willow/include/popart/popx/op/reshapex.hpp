// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESHAPEX_HPP
#define GUARD_NEURALNET_RESHAPEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ReshapeBaseOpx : public Opx {
public:
  ReshapeBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex inIndex, OutIndex outIndex) const final;
};

class ReshapeOpx : public ReshapeBaseOpx {
public:
  ReshapeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ReshapeInplaceOpx : public ReshapeBaseOpx {
public:
  ReshapeInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// The gradient of a reshape is the reshape in reverse
class ReshapeGradOpx : public ReshapeOpx {
public:
  ReshapeGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
