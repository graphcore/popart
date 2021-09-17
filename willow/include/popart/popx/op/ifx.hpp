// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IFX_HPP
#define GUARD_NEURALNET_IFX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class IfOpx : public PopOpx {
public:
  IfOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  void copyInputs(poplar::program::Sequence &thenProg,
                  poplar::program::Sequence &elseProg) const;

  void callBranch(poplar::program::Sequence &prog, const Graph &graph) const;

  void copyOutputs(poplar::program::Sequence &thenProg,
                   poplar::program::Sequence &elseProg,
                   const std::vector<snap::Tensor> &outputs) const;

  std::vector<snap::Tensor> prepareOutputs() const;

  std::vector<std::tuple<TensorId, TensorId, bool>>
  getInputsToPrepare() const override;
};

class IfGradOpx : public IfOpx {
public:
  IfGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
