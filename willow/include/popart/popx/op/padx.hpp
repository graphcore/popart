// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PADX_HPP
#define GUARD_NEURALNET_PADX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

class BasePadOp;

// With padding, there are 4 cases to consider,
//
//
//              downstream modifiers | no downstream modifiers
//                                   |
// PadInplace             A          |           B
// -------------------------------------------------------------
// PadOutplace            C          |           D
//                                   |
//
//                .....   downstream
//   xxx    pad   .xxx.    consumer
//   xxx    --->  .xxx.     --->
//   xxx          .xxx.
//                .....
//
//  PadOutplace (C, D above) promises that there is no aliaising between the
//  'x's is the pad Op's input and output.  WWith PadInplace the 'x's might
//  be aliased before and after.
//
// The padding values (the '.'s above) should be rewritten to their constant
// value (we're assuming constant padding here, not edge or reflect) every run
// of the Pad Op, otherwise if they are modified inplace they will be incorrect
// in subsequent iterations.
//
// Two options to ensure they are always correct:
//
// 1) make them poplar constant Tensors, which cannot (by definition) be
// modified by downstream consumers, even if the Ir claims that they are
// modified inplace.
//
// 2) run a program to set them to the constant value everytime that pad's grow
// function is called.
//
// We have oscillated between the 2 approaches, the current approach is 2. We
// ensure the padding is set to the correct value every iteration. See the
// member "cloneNcopyEdges".
//
//
namespace popx {

class BasePadOpx : public Opx {
public:
  BasePadOpx(Op *, Devicex *);
  const BasePadOp &getBasePadOp() const;
  poplar::Tensor padGrow(poplar::Tensor inTensor,
                         poplar::program::Sequence &) const;

private:
  poplar::Tensor constantModePadGrow(poplar::Tensor inTensor,
                                     poplar::program::Sequence &) const;

  // Padding with a constant needs to layout the constant. Sometimes there is an
  // obvious good choice for this: an example is if this Pad is a SliceGrad,
  // then the padding should have the layout of the original Tensor sliced.
  // TODO T22334 : generalize the search for propitious layout
  std::pair<bool, poplar::Tensor> getPropitiousPadLayout() const;

  // TODO T22336  : move this functionality to poplibs
  // Pad the tensor "toPad", so that it has the same shape as toMapEdgesFrom,
  // and the padding added has the same tile mapping as the corresponding
  // element of toMapEdgesFrom. toMapEdgesFrom must be of the correct shape, as
  // specified by the Ir PadOp.
  poplar::Tensor
  padWithTargetMapping(const poplar::Tensor &toPad,
                       const poplar::Tensor &toMapEdgesFrom) const;

  // Return a Tensor of the same shape as inTensor, which is an alias of
  // inTensor at the core, and a copy of inTensor on the padding edges.
  poplar::Tensor cloneNcopyEdges(poplar::Tensor inTensor,
                                 poplar::program::Sequence &) const;

  struct Chisseled {
    Chisseled(poplar::Tensor c,
              const std::vector<poplar::Tensor> &l,
              const std::vector<poplar::Tensor> &u)
        : core(c), lows(l), upps(u) {}
    poplar::Tensor core;
    std::vector<poplar::Tensor> lows;
    std::vector<poplar::Tensor> upps;
  };

  Chisseled getChisseled(const poplar::Tensor &) const;

  poplar::Tensor flip(const poplar::Tensor &) const;

  poplar::Tensor unflippedPadGrow(poplar::Tensor inTensor,
                                  poplar::program::Sequence &) const;
};

class PadOpx : public BasePadOpx {
public:
  PadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class PadInplaceOpx : public BasePadOpx {
public:
  PadInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
