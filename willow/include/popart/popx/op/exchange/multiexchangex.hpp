// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIEXCHANGEX_HPP
#define GUARD_NEURALNET_MULTIEXCHANGEX_HPP

#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class MultiExchangeOpx : public ExchangeBaseOpx {
public:
  MultiExchangeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  bool canUnwind(InIndex, OutIndex) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Graph &inGraph(InIndex in) const;
  snap::Graph &outGraph(OutIndex out) const;
};

} // namespace popx
} // namespace popart

#endif
