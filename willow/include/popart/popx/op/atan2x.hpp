// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2X_HPP
#define GUARD_NEURALNET_ATAN2X_HPP

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

class Atan2Computex : public EwbComputex {
public:
  explicit Atan2Computex(EwbComputex::InplacePolicy ip);

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

class Atan2Opx : public ElementWiseBinaryOutplaceOpx {
public:
  Atan2Opx(Op *, Devicex *);
};

class Atan2LhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  Atan2LhsInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
