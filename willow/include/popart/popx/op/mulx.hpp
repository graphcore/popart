// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULX_HPP
#define GUARD_NEURALNET_MULX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/op/reducesumx.hpp>

namespace popart {
namespace popx {

class MulComputex : public EwbComputex {
public:
  explicit MulComputex(EwbComputex::InplacePolicy ip);

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

class MulOpx : public ElementWiseBinaryOutplaceOpx {
public:
  MulOpx(Op *, Devicex *);
};

class MulLhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  MulLhsInplaceOpx(Op *, Devicex *);
};

class MulRhsInplaceOpx : public ElementWiseBinaryInplaceOpx {
public:
  MulRhsInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
