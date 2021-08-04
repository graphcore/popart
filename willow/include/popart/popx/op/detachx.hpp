// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DETACHX_HPP
#define GUARD_NEURALNET_DETACHX_HPP

#include <popart/names.hpp>
#include <popart/op/detach.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class DetachOpx : public ElementWiseUnaryOpx {

public:
  DetachOpx(popart::Op *, popart::popx::Devicex *);
  void grow(poplar::program::Sequence &) const;
};

class DetachInplaceOpx : public ElementWiseUnaryOpx {
public:
  DetachInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
