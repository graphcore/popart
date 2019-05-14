#ifndef GUARD_NEURALNET_CALLX_HPP
#define GUARD_NEURALNET_CALLX_HPP

#include <boost/optional.hpp>

#include <poponnx/popx/opx.hpp>

using boost::optional;

namespace poponnx {

namespace popx {

class CallOpx : public Opx {
public:
  CallOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  poplar::Tensor createInput(InIndex, const std::string &name) const;
  InputCreatorType getInputCreatorType(int) const;
  bool createsEquiv(int index0, const Opx *opx1, int index1) const;
  std::vector<TensorId> mustExistBeforeCreate(int index0) const;

private:
  // Copy aliased or modifed inputs back from graph.
  void copyModified(poplar::program::Sequence &prog) const;
  // Copy CallOp inputs to Graph input tensors.
  // If the graph input tensors have not been created,
  // they are created here by cloning the CallOp inputs.
  void copyInputs(poplar::program::Sequence &prog) const;
  // Copy the Graph output tensors to the CallOp outputs.
  void copyOutputs(poplar::program::Sequence &prog,
                   const std::vector<poplar::Tensor> &outputs) const;
  void doCall(poplar::program::Sequence &prog) const;
  // preparing outputs at returned (calling) site
  std::vector<poplar::Tensor> prepareOutputs() const;

  optional<InputCreatorCandidate> getCreator(InIndex) const;
};

} // namespace popx
} // namespace poponnx

#endif
