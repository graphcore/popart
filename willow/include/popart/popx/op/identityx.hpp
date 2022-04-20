// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class IdentityOpx : public ElementWiseUnaryOpx {
public:
  IdentityOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class IdentityInplaceOpx : public PopOpx {
public:
  IdentityInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class IdentityGradOpx : public ElementWiseUnaryOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class IdentityLossOpx : public PopOpx {
public:
  IdentityLossOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;

  snap::Tensor
  unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class IdentityLossGradOpx : public PopOpx {
public:
  IdentityLossGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  bool outputCreatedExternally(OutIndex) const final { return true; }
};

} // namespace popx
} // namespace popart

#endif
