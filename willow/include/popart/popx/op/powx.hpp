// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POWX_HPP
#define GUARD_NEURALNET_POWX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {
namespace popx {

class PowComputex : public EwbComputex {
public:
  explicit PowComputex(EwbComputex::InplacePolicy ip);

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
