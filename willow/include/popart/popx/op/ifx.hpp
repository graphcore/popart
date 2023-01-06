// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_IFX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_IFX_HPP_

#include <vector>
#include <popart/popx/opx.hpp>
#include <popart/popx/preparedtensor.hpp>

namespace poplar {
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Graph;
class Op;

namespace popx {
class Devicex;

class IfOpx : public Opx {
public:
  IfOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  void copyInputs(poplar::program::Sequence &thenProg,
                  poplar::program::Sequence &elseProg) const;

  void callBranch(poplar::program::Sequence &prog, const Graph &graph) const;

  void copyOutputs(poplar::program::Sequence &thenProg,
                   poplar::program::Sequence &elseProg,
                   const std::vector<poplar::Tensor> &outputs) const;

  std::vector<poplar::Tensor> prepareOutputs() const;

  PreparedTensorInfos getInputsToPrepare() const override;
};

class IfGradOpx : public IfOpx {
public:
  IfGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_IFX_HPP_
