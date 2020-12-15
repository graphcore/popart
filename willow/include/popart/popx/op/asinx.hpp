// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ASINX_HPP
#define GUARD_NEURALNET_ASINX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class AsinComputex : public EwuComputex {

public:
  AsinComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new AsinComputex);
  }
};

class AsinOpx : public ElementWiseUnaryOutplaceOpx {
public:
  AsinOpx(Op *, Devicex *);
};

class AsinInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  AsinInplaceOpx(Op *, Devicex *);
};

class AsinGradOpx : public Opx {
public:
  AsinGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
