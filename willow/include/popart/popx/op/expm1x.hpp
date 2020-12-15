// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPM1X_HPP
#define GUARD_NEURALNET_EXPM1X_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class Expm1Computex : public EwuComputex {

public:
  Expm1Computex() = default;

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
    return std::unique_ptr<EwuComputex>(new Expm1Computex());
  }
};

class Expm1Opx : public ElementWiseUnaryOutplaceOpx {
public:
  Expm1Opx(Op *, Devicex *);
};

class Expm1InplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  Expm1InplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
