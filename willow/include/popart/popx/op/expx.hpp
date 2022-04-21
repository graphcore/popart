// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPX_HPP
#define GUARD_NEURALNET_EXPX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class ExpComputex : public EwuComputex {

public:
  ExpComputex() = default;

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
    return std::unique_ptr<EwuComputex>(new ExpComputex);
  }
};

class ExpOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ExpOpx(Op *, Devicex *);
};

class ExpInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ExpInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
