// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ROUNDX_HPP
#define GUARD_NEURALNET_ROUNDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class RoundComputex : public EwuComputex {

public:
  RoundComputex() {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          snap::Graph &,
                          const poplar::Tensor &tensor,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new RoundComputex());
  }
};

class RoundOpx : public ElementWiseUnaryOutplaceOpx {
public:
  RoundOpx(Op *, Devicex *);
};

class RoundInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  RoundInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
