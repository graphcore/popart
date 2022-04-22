// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTSIGNX_HPP
#define GUARD_NEURALNET_SOFTSIGNX_HPP

#include "popart/popx/debugcontextx.hpp"
#include <string>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;
class Tensor;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class SoftSignComputex : public EwuComputex {
public:
  SoftSignComputex() {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;
};

class SoftSignOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SoftSignOpx(Op *, Devicex *);
};

class SoftSignInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SoftSignInplaceOpx(Op *, Devicex *);
};

class SoftSignGradOpx : public PopOpx {
public:
  SoftSignGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
