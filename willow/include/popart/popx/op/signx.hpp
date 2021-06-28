// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SIGNX_HPP
#define GUARD_NEURALNET_SIGNX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class SignComputex : public EwuComputex {

public:
  SignComputex() {}

  snap::Tensor outplace(poplar::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new SignComputex());
  }
};

class SignOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SignOpx(Op *, Devicex *);
};

class SignInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SignInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
