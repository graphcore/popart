// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATANX_HPP
#define GUARD_NEURALNET_ATANX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class AtanComputex : public EwuComputex {

public:
  AtanComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new AtanComputex);
  }
};

class AtanOpx : public ElementWiseUnaryOutplaceOpx {
public:
  AtanOpx(Op *, Devicex *);
};

class AtanInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  AtanInplaceOpx(Op *, Devicex *);
};

class AtanGradOpx : public PopOpx {
public:
  AtanGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
