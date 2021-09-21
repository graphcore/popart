// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALLX_HPP
#define GUARD_NEURALNET_CALLX_HPP

#include <popart/popx/op/subgraphx.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {
namespace popx {

class CallOpx : public SubgraphOpx {
public:
  CallOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  void grow(std::vector<poplar::program::Sequence> &) const final;
  InputCreatorType getInputCreatorType(InIndex) const;

private:
  // Copy aliased or modified inputs back from graph.
  void copyModified(poplar::program::Sequence &prog, InIndex inputIndex) const;
  // Copy CallOp inputs to Graph input tensors.
  // If the graph input tensors have not been created,
  // they are created here by cloning the CallOp inputs.
  void copyInput(poplar::program::Sequence &prog, InIndex inputIndex) const;
  // Copy the Graph output tensors to the CallOp outputs.
  void copyOutput(poplar::program::Sequence &prog, OutIndex outputIndex) const;
  // Call a specific subgraph part.
  void doCall(poplar::program::Sequence &prog,
              SubgraphPartIndex subgraphPart) const;
};

class CallGradOpx : public CallOpx {
public:
  CallGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
