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
// in subsequent iterations. However, with the current implementation, we do not
// need to reset the padding values, because they are poplar constant Tensors
// and therefore cannot be modified by downstream consumers, even if the Ir
// claims that they are modified inplace.
//
// Thus we do:
//
// A) nothing, no need to reset the .s as they are poplar constants.
// B) nothing.
// C) copy the xs, no need to reset the .s as they are poplar constants.
// D) copy the xs.
//
namespace popx {

class BasePadOpx : public Opx {
public:
  BasePadOpx(Op *, Devicex *);
  const BasePadOp &getBasePadOp() const;
  poplar::Tensor padGrow(poplar::Tensor inTensor,
                         poplar::program::Sequence &) const;

private:
  bool mustRewritePadding() const { return true; }
  poplar::Tensor constantModePadGrow(poplar::Tensor inTensor,
                                     poplar::program::Sequence &) const;

  // Padding with a constant needs to layout the constant. Sometimes there is an
  // obvious good choice for this: an example is if this Pad is a SliceGrad,
  // then the padding should have the layout of the original Tensor sliced.
  // TODO T22334 : generalize the search for propitious layout
  std::pair<bool, poplar::Tensor> getPropitiousPadLayout() const;

  // TODO T22336  : move this functionality to poplibs
  poplar::Tensor
  padWithTargetMapping(const poplar::Tensor &toPad,
                       const poplar::Tensor &toMapEdgesFrom) const;
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
