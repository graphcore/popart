// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_FLOORX_HPP
#define GUARD_NEURALNET_FLOORX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class FloorComputex : public EwuComputex {

public:
  FloorComputex() {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new FloorComputex());
  }
};

class FloorOpx : public ElementWiseUnaryOutplaceOpx {
public:
  FloorOpx(Op *, Devicex *);
};

class FloorInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  FloorInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
