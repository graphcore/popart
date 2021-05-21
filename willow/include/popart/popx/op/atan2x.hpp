// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2X_HPP
#define GUARD_NEURALNET_ATAN2X_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class Atan2Computex : public EwbComputex {
public:
  explicit Atan2Computex(EwbComputex::InplacePolicy ip);

  poplar::Tensor outplace(poplar::program::Sequence &,
                          snap::Graph &,
                          const poplar::Tensor &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
               const poplar::Tensor &,
               const poplar::Tensor &,
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
