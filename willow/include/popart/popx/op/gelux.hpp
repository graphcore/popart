// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GELUX_HPP
#define GUARD_NEURALNET_GELUX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class GeluComputex : public EwuComputex {
public:
  GeluComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          snap::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               snap::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new GeluComputex());
  }
};

class GeluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  GeluOpx(Op *, Devicex *);
};

class GeluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  GeluInplaceOpx(Op *, Devicex *);
};

class GeluGradOpx : public PopOpx {
public:
  GeluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
