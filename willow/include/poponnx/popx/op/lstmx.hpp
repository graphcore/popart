#ifndef GUARD_NEURALNET_LSTMX_HPP
#define GUARD_NEURALNET_LSTMX_HPP

#include <boost/optional.hpp>
#include <popnn/Lstm.hpp>
#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class LSTMOp;

namespace popx {

class LSTMOpx : public Opx {
public:
  LSTMOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  bool canCreateInput(InIndex index) const final;
  poplar::Tensor createInput(InIndex index) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const;

private:
  void growBias(poplar::program::Sequence &) const;
  LSTMOp *getLSTMOp() const;
  popnn::lstm::LstmParams createLSTMParams() const;
  popnn::lstm::LstmWeights getLSTMWeights() const;
  popnn::lstm::LstmState getInitialState() const;
  poplar::Tensor createLSTMInput() const;
  void prepareInitialState(popnn::lstm::LstmState &,
                           poplar::program::Sequence &) const;
  static poplar::Tensor reshapePoplibWeightsForOnnx(poplar::Tensor,
                                                    bool transpose);
  std::unique_ptr<poplar::Tensor> createIntermediate() const;

  mutable boost::optional<popnn::lstm::LstmWeights> weights;
  mutable boost::optional<popnn::lstm::LstmState> initial_state;
};

} // namespace popx
} // namespace poponnx

#endif
