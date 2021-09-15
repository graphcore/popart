// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIEXCHANGEX_HPP
#define GUARD_NEURALNET_MULTIEXCHANGEX_HPP

#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxstate.hpp>

namespace popart {

namespace popx {

class MultiExchangeOpx : public ExchangeBaseOpx {
public:
  MultiExchangeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  std::vector<std::pair<int, int>> getSegments() const;

  // Resolves the part ID by fist resolving which exchange descriptors consume
  // the input tensor, then returns the part IDs matching the descriptor index
  std::set<OpxGrowPartId> getInGrowPartIds(Tensor *inTensor) const final;

  // Resolves the part ID by fist resolving which exchange descriptors produce
  // the output tensor, then returns the part IDs matching the descriptor index
  OpxGrowPartId getOutGrowPartId(Tensor *outTensor) const final;

  // Each part ID 1:1 matches an exchange descriptor index
  void growPart(OpxGrowPartId id) const final;

  InputCreatorType getInputCreatorType(InIndex index) const final;
  bool canUnwind(InIndex, OutIndex) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Graph &inGraph(InIndex in) const;
  snap::Graph &outGraph(OutIndex out) const;
};

class MultiExchangeOpxState : public OpxState {
public:
  std::map<int, poplar::program::Sequence> preSeqs;
  std::map<int, poplar::program::Sequence> exchangeSeqs;
  std::map<int, poplar::program::Sequence> postSeqs;
};

} // namespace popx
} // namespace popart

#endif
