// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SIGNX_HPP
#define GUARD_NEURALNET_SIGNX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <popart/popx/op/elementwisex.hpp>

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SignComputex : public EwuComputex {

public:
  SignComputex() {}

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
