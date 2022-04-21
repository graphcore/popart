// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RELUX_HPP
#define GUARD_NEURALNET_RELUX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class ReluComputex : public EwuComputex {

public:
  ReluComputex() = default;

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new ReluComputex);
  }
};

class ReluOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ReluOpx(Op *, Devicex *);
};

class ReluInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ReluInplaceOpx(Op *, Devicex *);
};

class ReluGradOpx : public PopOpx {
public:
  ReluGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
