#ifndef GUARD_NEURALNET_CALLX_HPP
#define GUARD_NEURALNET_CALLX_HPP

#include <boost/optional.hpp>
#include <popart/popx/op/subgraphopx.hpp>
#include <popart/popx/opx.hpp>

using boost::optional;

namespace popart {

namespace popx {

class CallOpx : public SubgraphOpx {
public:
  CallOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  std::pair<std::vector<ICreatorCandidatePtr>, std::vector<UnwindEndpointPtr>>
      getEndpoints(InIndex, std::vector<OpxInAndOutIndex>) const;
  InputCreatorType getInputCreatorType(InIndex) const;

  std::vector<std::tuple<TensorId, TensorId, bool>>
  getOutputsToPrepare() const final;

private:
  // Copy aliased or modifed inputs back from graph.
  void copyModified(poplar::program::Sequence &prog) const;
  // Copy CallOp inputs to Graph input tensors.
  // If the graph input tensors have not been created,
  // they are created here by cloning the CallOp inputs.
  void copyInputs(poplar::program::Sequence &prog) const;
  // Copy the Graph output tensors to the CallOp outputs.
  void copyOutputs(poplar::program::Sequence &prog) const;
  void doCall(poplar::program::Sequence &prog) const;
  // preparing outputs at returned (calling) site
  std::vector<std::pair<poplar::Tensor, bool>> prepareOutputs() const;
};

} // namespace popx
} // namespace popart

#endif
