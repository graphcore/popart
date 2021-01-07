// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LSTMX_HPP
#define GUARD_NEURALNET_LSTMX_HPP

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

template <typename LSTMOP> class PopartLSTMOpxBase : public Opx {
public:
  PopartLSTMOpxBase(Op *op, Devicex *devicex) : Opx(op, devicex) {}

protected:
  popnn::lstm::LstmParams createLSTMParams() const {
    auto &lstmOp = getOp<LSTMOP>();
    auto inInfo  = lstmOp.inInfo(lstmOp.getInputInIndex());

    auto inputSize  = static_cast<unsigned>(lstmOp.getInputSize());
    auto seqLength  = static_cast<unsigned>(lstmOp.getSeqLength());
    auto batchSize  = static_cast<unsigned>(lstmOp.getBatchSize());
    auto hiddenSize = static_cast<unsigned>(lstmOp.getHiddenSize());

    auto params = popnn::lstm::LstmParams(
        popType(inInfo), batchSize, seqLength, {inputSize, hiddenSize});
    params.outputFullSequence = lstmOp.outputFullSequence;
    return params;
  }

  poplar::Tensor createBiasesInput() const {
    return popnn::lstm::createWeightsBiases(graph(),
                                            createLSTMParams(),
                                            debugContext("createWeights"),
                                            dv_p->lowering().lstmOptions,
                                            &dv_p->matmulCache);
  }

  poplar::Tensor getBiases(poplar::program::Sequence &prog) const {
    auto &lstmOp = getOp<LSTMOP>();

    if (hasInput(lstmOp.getBiasesInIndex())) {
      return getInTensor(lstmOp.getBiasesInIndex());
    } else {
      auto biases = createBiasesInput();
      popops::zero(graph(), biases, prog, debugContext("zeroBiases"));
      return biases;
    }
  }

  popnn::lstm::LstmState createInitialStateInput() const {
    return createInitialState(graph(),
                              createLSTMParams(),
                              debugContext("createInitialState"),
                              dv_p->lowering().lstmOptions,
                              &dv_p->matmulCache);
  }

  popnn::lstm::LstmState
  getInitialState(poplar::program::Sequence &prog) const {
    auto &lstmOp = getOp<LSTMOP>();

    if (hasInput(lstmOp.getInitialStateInIndex())) {
      auto initialState     = getInTensor(lstmOp.getInitialStateInIndex());
      auto initialOutput    = initialState.slice(0, 1).squeeze({0});
      auto initialCellState = initialState.slice(1, 2).squeeze({0});
      return {initialOutput, initialCellState};
    } else {
      auto initialState = createInitialStateInput();
      zeroInitialState(graph(), initialState, prog, debugContext());
      return initialState;
    }
  }

  popnn::lstm::LstmWeights getWeights(poplar::program::Sequence &prog) const {
    auto &lstmOp    = getOp<LSTMOP>();
    auto inputSize  = lstmOp.getInputSize();
    auto hiddenSize = lstmOp.getHiddenSize();

    auto weights = getInTensor(lstmOp.getWeightsInIndex());
    auto biases  = getBiases(prog);

    auto inputWeights  = weights.slice(0, inputSize, 1);
    auto outputWeights = weights.slice(inputSize, inputSize + hiddenSize, 1);
    popnn::lstm::LstmWeights lstmWeights = {
        inputWeights, outputWeights, biases};
    return lstmWeights;
  }
};

class PopartLSTMOpx : public PopartLSTMOpxBase<PopartLSTMOp> {
public:
  PopartLSTMOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const;

private:
  poplar::Tensor createLSTMInput() const;
  poplar::Tensor createWeightsInput() const;
  std::unique_ptr<poplar::Tensor> getIntermediates() const;
};

class PopartLSTMGradOpx : public PopartLSTMOpxBase<PopartLSTMGradOp> {
public:
  PopartLSTMGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
