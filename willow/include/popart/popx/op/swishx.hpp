// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SWISHX_HPP
#define GUARD_NEURALNET_SWISHX_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class SwishComputex : public EwuComputex {

public:
  SwishComputex() = default;

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new SwishComputex());
  }
};

class SwishOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SwishOpx(Op *, Devicex *);
};

class SwishInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SwishInplaceOpx(Op *, Devicex *);
};

class SwishGradOpx : public PopOpx {
public:
  SwishGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
