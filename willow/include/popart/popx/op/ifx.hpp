#ifndef GUARD_NEURALNET_IFX_HPP
#define GUARD_NEURALNET_IFX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

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
};

class IfGradOpx : public IfOpx {
public:
  IfGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
