// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CEILX_HPP
#define GUARD_NEURALNET_CEILX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class CeilComputex : public EwuComputex {

public:
  CeilComputex() {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new CeilComputex());
  }
};

class CeilOpx : public ElementWiseUnaryOutplaceOpx {
public:
  CeilOpx(Op *, Devicex *);
};

class CeilInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  CeilInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
