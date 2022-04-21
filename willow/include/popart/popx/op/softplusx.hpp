// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTPLUSX_HPP
#define GUARD_NEURALNET_SOFTPLUSX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class SoftPlusComputex : public EwuComputex {
public:
  SoftPlusComputex() {}

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;
};

class SoftPlusOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SoftPlusOpx(Op *, Devicex *);
};

class SoftPlusInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SoftPlusInplaceOpx(Op *, Devicex *);
};

class SoftPlusGradOpx : public PopOpx {
public:
  SoftPlusGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
