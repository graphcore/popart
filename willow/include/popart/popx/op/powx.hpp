// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POWX_HPP
#define GUARD_NEURALNET_POWX_HPP

#include "popart/popx/debugcontextx.hpp"
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

class PowComputex : public EwbComputex {
public:
  explicit PowComputex(EwbComputex::InplacePolicy ip);

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  snap::Tensor maybeInplace(snap::program::Sequence &,
                            snap::Graph &,
                            const snap::Tensor &,
                            const snap::Tensor &,
                            const poplar::DebugNameAndId &,
                            const std::string &) const final;
};

class PowOpx : public ElementWiseBinaryOutplaceOpx {
public:
  PowOpx(Op *, Devicex *);
};

class PowLhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  PowLhsInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
