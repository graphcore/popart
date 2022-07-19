// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CALLX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CALLX_HPP_

#include <vector>
#include <popart/popx/op/subgraphx.hpp>
#include <popart/popx/popopx.hpp>

#include "popart/names.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class CallOpx : public SubgraphOpx {
public:
  CallOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  void grow(std::vector<snap::program::Sequence> &) const final;
  InputCreatorType getInputCreatorType(InIndex) const;

private:
  // Copy aliased or modified inputs back from graph.
  void copyModified(snap::program::Sequence &prog, InIndex inputIndex) const;
  // Copy CallOp inputs to Graph input tensors.
  // If the graph input tensors have not been created,
  // they are created here by cloning the CallOp inputs.
  void copyInput(snap::program::Sequence &prog, InIndex inputIndex) const;
  // Copy the Graph output tensors to the CallOp outputs.
  void copyOutput(snap::program::Sequence &prog, OutIndex outputIndex) const;
  // Call a specific subgraph part.
  void doCall(snap::program::Sequence &prog,
              SubgraphPartIndex subgraphPart) const;
};

class CallGradOpx : public CallOpx {
public:
  CallGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CALLX_HPP_
