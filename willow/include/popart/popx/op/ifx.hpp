// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IFX_HPP
#define GUARD_NEURALNET_IFX_HPP

#include <vector>
#include <popart/popx/popopx.hpp>
#include <popart/popx/preparedtensor.hpp>

namespace snap {
class Tensor;
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Graph;
class Op;

namespace popx {
class Devicex;

class IfOpx : public PopOpx {
public:
  IfOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  void copyInputs(snap::program::Sequence &thenProg,
                  snap::program::Sequence &elseProg) const;

  void callBranch(snap::program::Sequence &prog, const Graph &graph) const;

  void copyOutputs(snap::program::Sequence &thenProg,
                   snap::program::Sequence &elseProg,
                   const std::vector<snap::Tensor> &outputs) const;

  std::vector<snap::Tensor> prepareOutputs() const;

  PreparedTensorInfos getInputsToPrepare() const override;
};

class IfGradOpx : public IfOpx {
public:
  IfGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
