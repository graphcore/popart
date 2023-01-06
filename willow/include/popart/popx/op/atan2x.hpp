// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATAN2X_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATAN2X_HPP_

#include <string>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class Atan2Computex : public EwbComputex {
public:
  explicit Atan2Computex(EwbComputex::InplacePolicy ip);

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  poplar::Tensor maybeInplace(poplar::program::Sequence &,
                              poplar::Graph &,
                              poplar::Tensor &,
                              poplar::Tensor &,
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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ATAN2X_HPP_
