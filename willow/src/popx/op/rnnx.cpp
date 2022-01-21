// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/rnn.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/lstmxutil.hpp>
#include <popart/popx/op/rnnx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

RNNOpx::RNNOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<RNNOp>(op, {Onnx::Operators::RNN_7});
}

void RNNOpx::grow(snap::program::Sequence &prog) const {
  // sets up bias tensor
  auto bias = getBias(prog);

  // sets up initialH tensor
  auto initialH = getInitialH(prog);

  // Fetch input tensors
  auto X = getInTensor(RNNOp::getInputInIndex()).getPoplarTensor();
  auto W = getInTensor(RNNOp::getInputWeightsInIndex()).getPoplarTensor();
  auto R = getInTensor(RNNOp::getRecurrenceWeightsInIndex()).getPoplarTensor();

  auto &rnn_op        = getOp<RNNOp>();
  auto activation     = convert(rnn_op.activation_attribute);
  unsigned inputSize  = rnn_op.getInputSize();
  unsigned seqLen     = rnn_op.getSeqLength();
  unsigned hiddenSize = rnn_op.getHiddenSize();
  unsigned batchSize  = rnn_op.getBatchSize();
  const poplar::Type elem_type =
      popType(rnn_op.inInfo(RNNOp::getInputInIndex()));
  auto output = graph().addVariable(elem_type,
                                    {seqLen, 1, batchSize, hiddenSize},
                                    poplar::VariableMappingMethod::LINEAR,
                                    debugContext("rnn/output"));
  for (int i = 0; i < seqLen; i++) {

    snap::Tensor Hprev;
    if (i == 0) {
      Hprev = initialH[0];
    } else {
      Hprev = output[i - 1][0];
    }

    for (int j = 0; j < batchSize; j++) {
      // Wx = W * x
      auto Wx = poplin::matMul(graph().getPoplarGraph(),
                               W[0],
                               X[i][j].reshape({inputSize, 1}),
                               prog.getPoplarSequence(),
                               debugContext("rnn/Wx"))
                    .reshape({hiddenSize});

      // Rh = R * h
      auto Rh =
          poplin::matMul(graph().getPoplarGraph(),
                         R[0],
                         Hprev[j].reshape({hiddenSize, 1}).getPoplarTensor(),
                         prog.getPoplarSequence(),
                         debugContext("rnn/Rh"))
              .reshape({hiddenSize});

      // output[i][0][j] = Rh + Wx + b
      prog.add(snap::program::Copy(
          snap::Tensor{popops::add(graph().getPoplarGraph(),
                                   popops::add(graph().getPoplarGraph(),
                                               Wx,
                                               Rh,
                                               prog.getPoplarSequence(),
                                               debugContext("rnn/Wx+Rh")),
                                   bias.getPoplarTensor()[0],
                                   prog.getPoplarSequence(),
                                   debugContext("rnn/Wx+Rh+b")),
                       graph()},
          output[i][0][j],
          false,
          debugContext("rnn/output[i][0][j]")));
    }
    popnn::nonLinearityInPlace(graph().getPoplarGraph(),
                               activation,
                               output[i].getPoplarTensor(),
                               prog.getPoplarSequence(),
                               debugContext("rnn/activation"));
  }

  // set outputs
  snap::Tensor output_last = output[seqLen - 1];
  if (getOp<RNNOp>().hasOutput(RNNOp::getFullOutputOutIndex())) {
    setOutTensor(RNNOp::getFullOutputOutIndex(), output);
  }
  if (getOp<RNNOp>().hasOutput(RNNOp::getLastOutputOutIndex())) {
    setOutTensor(RNNOp::getLastOutputOutIndex(), cloneNcopy(prog, output_last));
  }
}

snap::Tensor RNNOpx::getBias(snap::program::Sequence &prog) const {
  snap::Tensor bias;
  auto &rnn_op         = getOp<RNNOp>();
  unsigned hidden_size = rnn_op.getHiddenSize();
  // default to a 0 tensor if bias not provided by user
  if (!rnn_op.hasBiasInput()) {
    const poplar::Type elem_type =
        popType(rnn_op.inInfo(RNNOp::getInputInIndex()));
    bias = getZerosTensor({1, hidden_size}, elem_type, "rnn/zero_bias");
  } else {
    // ONNX format is [1, 2 * hidden_size]
    auto bias_input = getInTensor(RNNOp::getBiasInIndex())
                          .getPoplarTensor()
                          .reshape({2, 1, hidden_size});

    // Add the biases up and store in bias tensor
    auto input_bias     = bias_input[0];
    auto recurrent_bias = bias_input[1];
    bias                = snap::Tensor{popops::add(graph().getPoplarGraph(),
                                    input_bias,
                                    recurrent_bias,
                                    prog.getPoplarSequence(),
                                    debugContext("rnn/add_bias")),
                        graph()};
  }
  return bias;
}

snap::Tensor RNNOpx::getInitialH(snap::program::Sequence &prog) const {
  auto &rnn_op = getOp<RNNOp>();
  snap::Tensor initialH;
  // default to a 0 tensor if initial_H not provided by user
  if (!rnn_op.hasInitialHInput()) {
    unsigned hidden_size = rnn_op.getHiddenSize();
    unsigned batch_size  = rnn_op.getBatchSize();
    const poplar::Type elem_type =
        popType(rnn_op.inInfo(RNNOp::getInputInIndex()));
    initialH = getZerosTensor(
        {1, batch_size, hidden_size}, elem_type, "rnn/zero_initialH");
  } else {
    initialH = getInTensor(RNNOp::getInitialHInIndex());
  }
  return initialH;
}

std::set<TensorId> RNNOpx::mustExistBeforeCreate(InIndex) const { return {}; }

RNNGradOpx::RNNGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<RNNGradOp>(op, Onnx::GradOperators::RNNGrad);
}

void RNNGradOpx::grow(snap::program::Sequence &prog) const {
  auto &rnn_grad_op    = getOp<RNNGradOp>();
  auto activation      = convert(rnn_grad_op.activation_attribute);
  unsigned batch_size  = rnn_grad_op.batch_size;
  unsigned hidden_size = rnn_grad_op.hidden_size;
  unsigned seq_length  = rnn_grad_op.seq_length;
  unsigned input_size  = rnn_grad_op.input_size;
  const poplar::Type elem_type =
      popType(rnn_grad_op.inInfo(RNNGradOp::getInputInIndex()));
  // get input tensors of forward op
  poplar::Tensor forward_input =
      getInTensor(RNNGradOp::getInputInIndex()).getPoplarTensor();
  poplar::Tensor forward_output =
      getInTensor(RNNGradOp::getFullOutputInIndex()).getPoplarTensor();
  poplar::Tensor R =
      getInTensor(RNNGradOp::getRecurrenceWeightsInIndex()).getPoplarTensor();
  poplar::Tensor W =
      getInTensor(RNNGradOp::getInputWeightsInIndex()).getPoplarTensor();
  poplar::Tensor initialH_input;
  if (rnn_grad_op.hasInitialHInput) {
    initialH_input =
        getInTensor(RNNGradOp::getInitialHInIndex()).getPoplarTensor();
  } else {
    initialH_input = getZerosTensor({1, batch_size, hidden_size},
                                    elem_type,
                                    "rnngrad/zero_initialH_input")
                         .getPoplarTensor();
  }

  // get output grads and combine them
  poplar::Tensor output_full_grad = getFullOutputGrad(prog).getPoplarTensor();
  poplar::Tensor output_last_grad = getLastOutputGrad(prog).getPoplarTensor();
  popops::addInPlace(graph().getPoplarGraph(),
                     output_full_grad[output_full_grad.dim(0) - 1],
                     output_last_grad,
                     prog.getPoplarSequence(),
                     debugContext("rnngrad/combine_output_grads"));
  // Crete tensors for intermediate and output grads

  // combined gradient of the output and further iterations of the RNN
  // dh[i] combines output[i] and dh[i+1]
  poplar::Tensor dh =
      cloneNcopy(
          prog,
          getZerosTensor(forward_output.shape(), elem_type, "rnngrad/zero_dh"))
          .getPoplarTensor();

  poplar::Tensor input_grad =
      cloneNcopy(prog,
                 getZerosTensor(forward_input.shape(),
                                elem_type,
                                "rnngrad/zero_input_grad"))
          .getPoplarTensor();
  poplar::Tensor input_weights_grad =
      cloneNcopy(prog,
                 getZerosTensor(
                     W.shape(), elem_type, "rnngrad/zero_input_weights_grad"))
          .getPoplarTensor();
  poplar::Tensor recurrence_weights_grad =
      cloneNcopy(prog,
                 getZerosTensor(R.shape(),
                                elem_type,
                                "rnngrad/zero_recurrence_weights_grad"))
          .getPoplarTensor();
  // bias_grad contains grad for one bias element, not both, because the grads
  // are identical.
  poplar::Tensor bias_grad;
  // no need to initialize the grad tensor if we won't use it
  if (rnn_grad_op.hasBiasInput) {
    // initialize bias_grad tensor
    // Tensor has size of a single bias element, not both
    bias_grad =
        cloneNcopy(prog,
                   getZerosTensor(
                       {1, hidden_size}, elem_type, "rnngrad/zero_bias_grad"))
            .getPoplarTensor();
  }
  poplar::Tensor initialH_grad;
  // no need to initialize the grad tensor if we won't use it
  if (rnn_grad_op.hasInitialHInput) {
    // initialize initialH_grad
    initialH_grad = cloneNcopy(prog,
                               getZerosTensor(initialH_input.shape(),
                                              elem_type,
                                              "rnngrad/zero_initialH_grad"))
                        .getPoplarTensor();
  }

  // For convenience denote:
  //   dL/dh[seqLen] = 0
  // The backwards pass for the recurrence function then becomes
  //   dL/dh[i] = dh[i+1]/dh[i] * dL/dh[i+1] + do[i]/dh[i] * dL/do[i]
  // Here:
  //   dL/do[i] = d_output[i] - combined gradients of both outputs.
  //   dL/dh[i+1] = dh[i+1] - calculated gradient of the next layer.
  //   do[i]/dh[i] = 1 - gradient of output with respect to current hidden
  //   layer.
  //     Note that output[i] = h[i].
  //   dh[i+1]/dh[i] = dh[i+1]/da[i+1] * da[i+1]/dh[i]
  //     Here a represents the output of the RNN equation before an activation
  //     is applied, and da is it's gradient
  //   dh[i+1]/da[i+1] = activation function gradient
  //   da[i+1]/dh[i] = R
  //   da[i+1]/db = 1
  //   da[i+1]/dx[i+1] = W
  //   da[i+1]/dW = x[i+1]
  //   da[i+1]/dR = h[i]

  // set last layer of dh
  prog.getPoplarSequence().add(
      poplar::program::Copy(output_full_grad[seq_length - 1],
                            dh[seq_length - 1],
                            false,
                            debugContext("rnngrad/copy_dh_last")));

  // Calculate all subsequent layers of dh backwards
  // This unrolls all but the first iteration, as the first one uses initial_H
  for (int i = seq_length - 1; i >= 0; i--) {
    for (int j = 0; j < batch_size; j++) {
      // Gradient back through nonlinearity
      poplar::Tensor da =
          popnn::nonLinearityInputGradient(graph().getPoplarGraph(),
                                           activation,
                                           forward_output[i][0][j],
                                           dh[i][0][j],
                                           prog.getPoplarSequence(),
                                           debugContext("rnngrad/da"));

      if (i != 0) {
        // dh[i-1][0][j] = R[0]^T * da + output_full_grad[i][0][j]
        prog.getPoplarSequence().add(poplar::program::Copy(
            popops::add(graph().getPoplarGraph(),
                        output_full_grad[i - 1][0][j],
                        poplin::matMul(graph().getPoplarGraph(),
                                       R[0].transpose(),
                                       da.reshape({hidden_size, 1}),
                                       prog.getPoplarSequence(),
                                       debugContext("rnngrad/R*da"))
                            .reshape({hidden_size}),
                        prog.getPoplarSequence(),
                        debugContext("rnngrad/R*da+output_full_grad")),
            dh[i - 1][0][j],
            false,
            debugContext("rnngrad/copy_dh")));
      } else if (rnn_grad_op.hasInitialHInput) {
        // initialH_grad[0][j] = R[0]^T * da
        prog.getPoplarSequence().add(
            poplar::program::Copy(poplin::matMul(graph().getPoplarGraph(),
                                                 R[0].transpose(),
                                                 da.reshape({hidden_size, 1}),
                                                 prog.getPoplarSequence(),
                                                 debugContext("rnngrad/da*R_2"))
                                      .reshape({hidden_size}),
                                  initialH_grad[0][j],
                                  false,
                                  debugContext("rnngrad/copy_initialH_grad")));
      }

      // input_grad[i][j] = W[0]^T * da
      prog.getPoplarSequence().add(
          poplar::program::Copy(poplin::matMul(graph().getPoplarGraph(),
                                               W[0].transpose(),
                                               da.reshape({hidden_size, 1}),
                                               prog.getPoplarSequence(),
                                               debugContext("rnngrad/W*da"))
                                    .reshape({input_size}),
                                input_grad[i][j],
                                false,
                                debugContext("rnngrad/copy_input_grad")));

      // input_weights_grad[0] += da * forward_input[i][j]^T
      popops::addInPlace(
          graph().getPoplarGraph(),
          input_weights_grad[0], // hidden_size x input_size
          poplin::matMul(graph().getPoplarGraph(),
                         da.reshape({hidden_size, 1}),
                         forward_input[i][j].reshape({1, input_size}),
                         prog.getPoplarSequence(),
                         debugContext("rnngrad/da*forward_input")),
          prog.getPoplarSequence(),
          debugContext("rnngrad/addTo_input_weights_grad"));

      // recurrence_weights_grad[0] += da * forward_output[i-1][0][j]^T
      popops::addInPlace(
          graph().getPoplarGraph(),
          recurrence_weights_grad[0], // hidden_size x hidden_size
          poplin::matMul(
              graph().getPoplarGraph(),
              da.reshape({hidden_size, 1}),
              (i == 0 ? initialH_input[0][j] : forward_output[i - 1][0][j])
                  .reshape({1, hidden_size}),
              prog.getPoplarSequence(),
              debugContext("rnngrad/da*forward_output")),
          prog.getPoplarSequence(),
          debugContext("rnngrad/addTo_recurrence_weights_grad"));

      if (rnn_grad_op.hasBiasInput) {
        // bias_grad[0] += da
        popops::addInPlace(graph().getPoplarGraph(),
                           bias_grad[0], // hidden_size
                           da,           // hidden_size
                           prog.getPoplarSequence(),
                           debugContext("rnngrad/addTo_bias_grad"));
      }
    }
  }

  // set input gradient
  setOutTensor(RNNGradOp::getInputOutIndex(),
               snap::Tensor{input_grad, graph()});

  // set input weights gradient
  setOutTensor(RNNGradOp::getInputWeightsOutIndex(),
               snap::Tensor{input_weights_grad, graph()});

  // set hidden state weights gradient
  setOutTensor(RNNGradOp::getRecurrenceWeightsOutIndex(),
               snap::Tensor{recurrence_weights_grad, graph()});

  // set biases gradient
  if (rnn_grad_op.hasBiasInput) {
    // propagate same gradient to both input and hidden bias
    setOutTensor(RNNGradOp::getBiasOutIndex(),
                 snap::Tensor{bias_grad.broadcast(2, 1), graph()});
  }

  // set initialH gradient
  if (rnn_grad_op.hasInitialHInput) {
    setOutTensor(RNNGradOp::getInitialHOutIndex(),
                 snap::Tensor{initialH_grad, graph()});
  }
}

snap::Tensor
RNNGradOpx::getLastOutputGrad(snap::program::Sequence &prog) const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  if (rnn_grad_op.hasLastOutputGradInput()) {
    return getInTensor(RNNGradOp::getLastOutputGradInIndex());
  } else {
    unsigned batch_size  = rnn_grad_op.batch_size;
    unsigned hidden_size = rnn_grad_op.hidden_size;
    const poplar::Type elem_type =
        popType(rnn_grad_op.inInfo(RNNGradOp::getInputInIndex()));
    auto zero = getZerosTensor({1, batch_size, hidden_size},
                               elem_type,
                               "rnngrad/zero_getLastOutputGrad");
    return zero;
  }
}

snap::Tensor
RNNGradOpx::getFullOutputGrad(snap::program::Sequence &prog) const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  if (rnn_grad_op.hasFullOutputGradInput()) {
    auto output_full_grad = getInTensor(RNNGradOp::getFullOutputGradInIndex());
    return cloneNcopy(prog, output_full_grad);
  } else {
    const poplar::Type elem_type =
        popType(rnn_grad_op.inInfo(RNNGradOp::getInputInIndex()));
    unsigned batch_size  = rnn_grad_op.batch_size;
    unsigned hidden_size = rnn_grad_op.hidden_size;
    unsigned seq_length  = rnn_grad_op.seq_length;

    return cloneNcopy(prog,
                      getZerosTensor({seq_length, 1, batch_size, hidden_size},
                                     elem_type,
                                     "rnngrad/zero_getFullOutputGrad"));
  }
}

namespace {
OpxCreator<RNNOpx> rnnOpxCreator({Onnx::Operators::RNN_7});
OpxCreator<RNNGradOpx> rnnGradOpxCreator(Onnx::GradOperators::RNNGrad);
} // namespace

} // namespace popx
} // namespace popart
