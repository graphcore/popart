// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IDENTITYX_HPP
#define GUARD_NEURALNET_IDENTITYX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class IdentityComputex : public EwuComputex {
public:
  IdentityComputex() {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new IdentityComputex());
  }
};

class IdentityOpx : public ElementWiseUnaryOutplaceOpx {
public:
  IdentityOpx(Op *, Devicex *);
};

class IdentityInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  IdentityInplaceOpx(Op *, Devicex *);
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
