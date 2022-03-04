// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATAN2X_HPP
#define GUARD_NEURALNET_ATAN2X_HPP

#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class Atan2Computex : public EwbComputex {
public:
  explicit Atan2Computex(EwbComputex::InplacePolicy ip);

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
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
