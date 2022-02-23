// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RNNX_HPP
#define GUARD_NEURALNET_RNNX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

class RNNOp;

namespace popx {

class RNNOpx : public PopOpx {
public:
  RNNOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

private:
  // Return the tensor type of any input tensor (they all have the same type)
  poplar::Type getTensorType() const;
  // Return minimum number of elements of type getTensorType(), so that they
  // take up 16 bytes in total for performance reasons
  unsigned getMinGrainSize() const;
  // Return sum of bias tensors, or 0 if bias tensors not provided by user
  snap::Tensor getBias(snap::program::Sequence &) const;
  // Return initialH, or 0 if initialH is not provided by user
  snap::Tensor getInitialH(snap::program::Sequence &) const;
  // Return program for a single step of the forward pass
  snap::program::Sequence getFwdStepProg(snap::Tensor &bias,
                                         snap::Tensor &initialH,
                                         snap::Tensor &output,
                                         snap::Tensor &H_prev) const;
};

class RNNGradOpx : public PopOpx {
public:
  RNNGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const;

private:
  // Return the tensor type of any forward input tensor (they all have the same
  // type)
  poplar::Type getTensorType() const;
  // Return minimum number of elements of type getTensorType(), so that they
  // take up 16 bytes in total for performance reasons
  unsigned getMinGrainSize() const;
  // Return last_output_grad, or 0 if the input does not exist
  snap::Tensor getLastOutputGrad() const;
  // Return full_output_grad, or 0 if the input does not exist
  snap::Tensor getFullOutputGrad() const;
  // Return program for a single step of the backwards pass
  snap::program::Sequence getBwdStepProg(snap::Tensor &input_grad,
                                         snap::Tensor &input_weights_grad,
                                         snap::Tensor &recurrence_weights_grad,
                                         snap::Tensor &bias_grad,
                                         snap::Tensor &dh,
                                         snap::Tensor &forward_output,
                                         snap::Tensor &forward_output_prev,
                                         snap::Tensor &full_output_grad) const;
};

} // namespace popx
} // namespace popart

#endif
