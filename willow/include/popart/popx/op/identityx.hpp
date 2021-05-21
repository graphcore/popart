// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class IdentityOpx : public ElementWiseUnaryOpx {
public:
  IdentityOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityInplaceOpx : public PopOpx {
public:
  IdentityInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityGradOpx : public ElementWiseUnaryOpx {
public:
  IdentityGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class IdentityLossOpx : public PopOpx {
public:
  IdentityLossOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

class IdentityLossGradOpx : public PopOpx {
public:
  IdentityLossGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  bool outputCreatedExternally(OutIndex) const final { return true; }
};

} // namespace popx
} // namespace popart

#endif
