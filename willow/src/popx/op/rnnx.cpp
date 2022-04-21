// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>

#include <snap/poputil/TileMapping.hpp>
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

#include <snap/popops/ElementWise.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

RNNOpx::RNNOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<RNNOp>(op, {Onnx::Operators::RNN_7});
}

void RNNOpx::grow(snap::program::Sequence &prog) const {
  // Sets up bias tensor
  auto bias = getBias(prog);

  // Sets up initialH tensor
  auto initialH = getInitialH(prog);

  auto &rnn_op                 = getOp<RNNOp>();
  unsigned max_seq_length      = rnn_op.getMaxSeqLength();
  unsigned hidden_size         = rnn_op.getHiddenSize();
  unsigned batch_size          = rnn_op.getBatchSize();
  unsigned num_directions      = rnn_op.getNumDirections();
  const poplar::Type elem_type = getTensorType();
  unsigned min_grain_size      = getMinGrainSize();

  // full_hidden_state output variable
  // Make it efficient to be sliced along the max_seq_length dimension
  auto output = snap::Tensor{
      popops::createSliceableTensor(
          graph().getPoplarGraph(),
          elem_type,
          {max_seq_length, num_directions, batch_size, hidden_size}, // shape
          {0},                                                       // dims
          {1},                                                       // sizes
          min_grain_size,
          debugContext("created_output_tensor")),
      graph()};
  graph().getPoplarGraph().setTileMapping(
      initialH.getPoplarTensor(),
      graph().getPoplarGraph().getTileMapping(output.getPoplarTensor()[0]));
  // graph().getPoplarGraph().setTileMapping(bias.getPoplarTensor(),
  // graph().getPoplarGraph().getTileMapping(initialH[0][0].getPoplarTensor()));
  // Create a variable to store the hidden_state value of previous iteration
  snap::Tensor H_prev = cloneNcopy(prog, initialH, "H_prev");

  // Set up loop index
  poplar::Tensor index =
      getScalarVariable(poplar::UNSIGNED_INT, "fwd_pass_index")
          .getPoplarTensor()
          .reshape({1});
  // i = 0 (counting upwards to max_seq_length - 1)
  popops::fill(graph().getPoplarGraph(),
               index,
               prog.getPoplarSequence(),
               0,
               debugContext("initialise_index_to_0"));
  // Get a program that performs a single iteration of the RNN recurrence.
  auto fwdStepProg = getFwdStepProg(bias, initialH, output, H_prev, index);

  // Repeat fwdStepProg max_seq_length times
  prog.add(snap::program::Repeat(
      max_seq_length, fwdStepProg, debugContext("repeat_forward_pass_step")));

  // Set outputs
  snap::Tensor output_last = output[max_seq_length - 1];
  if (getOp<RNNOp>().hasOutput(RNNOp::getFullHiddenStateOutIndex())) {
    setOutTensor(RNNOp::getFullHiddenStateOutIndex(), output);
  }
  if (getOp<RNNOp>().hasOutput(RNNOp::getLastHiddenStateOutIndex())) {
    setOutTensor(RNNOp::getLastHiddenStateOutIndex(),
                 cloneNcopy(prog, output_last));
  }
}

snap::program::Sequence RNNOpx::getFwdStepProg(snap::Tensor &bias,
                                               snap::Tensor &initialH,
                                               snap::Tensor &output,
                                               snap::Tensor &H_prev,
                                               poplar::Tensor &index) const {
  // Fetch remaining input tensors
  auto X = getInTensor(RNNOp::getInputInIndex()).getPoplarTensor();
  auto W = getInTensor(RNNOp::getInputWeightsInIndex()).getPoplarTensor();
  auto R = getInTensor(RNNOp::getRecurrenceWeightsInIndex()).getPoplarTensor();

  auto &rnn_op                 = getOp<RNNOp>();
  auto activation              = convert(rnn_op.activation_attribute);
  auto cache                   = &dv_p->matmulCache;
  const poplar::Type elem_type = getTensorType();

  // Sequence to be returned that contains the single iteration code
  snap::program::Sequence fwdStepProg(debugContext("rnn_fwd_sequence"),
                                      graph());

  // Input value to be worked on in this iteration
  poplar::Tensor X_slice = popops::dynamicSlice(graph().getPoplarGraph(),
                                                X,
                                                index,
                                                {0},
                                                {1},
                                                fwdStepProg.getPoplarSequence(),
                                                debugContext("X_slice"))[0];
  // batch_size x input_size, batch_size x hidden_size -> batch_size x
  // (input_size + hidden_size)
  auto XH = poplar::concat(X_slice, H_prev[0].getPoplarTensor(), 1);
  // hidden_size x input_size, hidden_size x hidden_size -> hidden_size x
  // (input_size + hidden_size)
  auto WR = poplar::concat(W[0], R[0], 1);
  // X[i]*W + h[i]*R
  auto XW_HR  = poplin::matMul(graph().getPoplarGraph(),
                              XH,
                              WR.transpose(),
                              fwdStepProg.getPoplarSequence(),
                              debugContext("XW+HR"),
                              {},
                              cache);
  auto H_next = XW_HR;
  if (rnn_op.hasBiasesInput()) {
    // H_next = XH + WR + b
    H_next = popops::add(graph().getPoplarGraph(),
                         XW_HR,
                         bias.getPoplarTensor(),
                         fwdStepProg.getPoplarSequence(),
                         debugContext("XW+HR+b"));
  }
  H_next = H_next.expand({0});
  // Apply nonlinearity
  popnn::nonLinearityInPlace(graph().getPoplarGraph(),
                             activation,
                             H_next,
                             fwdStepProg.getPoplarSequence(),
                             debugContext("activation"));
  H_next =
      popops::rearrange::regroupIfBeneficial(graph().getPoplarGraph(),
                                             H_next,
                                             H_prev.getPoplarTensor(),
                                             fwdStepProg.getPoplarSequence(),
                                             debugContext("regrouped_H_next"));
  // Copy H_next to H_prev for next iteration
  fwdStepProg.add(snap::program::Copy(snap::Tensor{H_next, graph()},
                                      H_prev,
                                      false,
                                      debugContext("H_next_copy_to_H_prev")));

  // Copy current hidden_state slice to output tensor
  popops::dynamicUpdate(graph().getPoplarGraph(),
                        output.getPoplarTensor(),
                        H_next.expand({0}),
                        index,
                        {0},
                        {1},
                        fwdStepProg.getPoplarSequence());

  // Increment index
  popops::addInPlace<unsigned int>(graph().getPoplarGraph(),
                                   index,
                                   1,
                                   fwdStepProg.getPoplarSequence(),
                                   debugContext("increment_fwd_index"));
  return fwdStepProg;
}

poplar::Type RNNOpx::getTensorType() const {
  auto &rnn_op = getOp<RNNOp>();
  return popType(rnn_op.inInfo(RNNOp::getInputInIndex()));
}

unsigned RNNOpx::getMinGrainSize() const {
  auto &rnn_op = getOp<RNNOp>();
  return std::max(
      1,
      16 / rnn_op.inInfo(RNNOp::getInputInIndex()).getDataTypeInfo()->nbytes());
}

snap::Tensor RNNOpx::getBias(snap::program::Sequence &prog) const {
  snap::Tensor combined_bias;
  auto &rnn_op            = getOp<RNNOp>();
  unsigned num_directions = rnn_op.getNumDirections();
  unsigned hidden_size    = rnn_op.getHiddenSize();
  if (!rnn_op.hasBiasesInput()) {
    // Default to a 0 tensor if bias not provided by user
    const poplar::Type elem_type = getTensorType();
    combined_bias =
        getZerosTensor({num_directions, hidden_size}, elem_type, "zero_bias");
  } else {
    // ONNX format is [num_directions, 2 * hidden_size]
    auto bias_input = getInTensor(RNNOp::getBiasesInIndex())
                          .getPoplarTensor()
                          .reshape({2, num_directions, hidden_size});

    // Return sum of biases
    auto input_bias     = bias_input[0];
    auto recurrent_bias = bias_input[1];
    combined_bias       = snap::Tensor{popops::add(graph().getPoplarGraph(),
                                             input_bias,
                                             recurrent_bias,
                                             prog.getPoplarSequence(),
                                             debugContext("combined_bias")),
                                 graph()};
  }
  return combined_bias;
}

snap::Tensor RNNOpx::getInitialH(snap::program::Sequence &prog) const {
  auto &rnn_op = getOp<RNNOp>();
  snap::Tensor initialH;
  unsigned num_directions = rnn_op.getNumDirections();
  if (!rnn_op.hasInitialHInput()) {
    // Default to a 0 tensor if initial_H not provided by user
    unsigned hidden_size         = rnn_op.getHiddenSize();
    unsigned batch_size          = rnn_op.getBatchSize();
    const poplar::Type elem_type = getTensorType();
    initialH =
        cloneNcopy(prog,
                   getZerosTensor({num_directions, batch_size, hidden_size},
                                  elem_type,
                                  "zero_initialH"));
  } else {
    initialH = getInTensor(RNNOp::getInitialHInIndex());
  }
  return initialH;
}

InputCreatorType RNNOpx::getInputCreatorType(InIndex index) const {
  if (index == RNNOp::getInputInIndex() ||
      index == RNNOp::getInputWeightsInIndex() ||
      index == RNNOp::getRecurrenceWeightsInIndex() ||
      index == RNNOp::getBiasesInIndex() ||
      index == RNNOp::getInitialHInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

snap::Tensor
RNNOpx::createInputTensor(InIndex index,
                          const poplar::DebugNameAndId &dnai) const {
  auto &rnn_op                 = getOp<RNNOp>();
  const poplar::Type elem_type = getTensorType();
  auto cache                   = &dv_p->matmulCache;
  unsigned max_seq_length      = rnn_op.getMaxSeqLength();
  unsigned batch_size          = rnn_op.getBatchSize();
  unsigned hidden_size         = rnn_op.getHiddenSize();
  unsigned input_size          = rnn_op.getInputSize();
  unsigned num_directions      = rnn_op.getNumDirections();
  unsigned min_grain_size      = getMinGrainSize();
  if (index == RNNOp::getInputInIndex()) {
    // We want to parallelize over batch_size and input_size dimensions
    // We can't parallelize over max_seq_length dimension, so we make sure
    // the tensor can be efficiently sliced along that dimension
    return snap::Tensor{popops::createSliceableTensor(
                            graph().getPoplarGraph(),
                            elem_type,
                            {max_seq_length, batch_size, input_size}, // shape
                            {0},                                      // dims
                            {1},                                      // sizes
                            min_grain_size,
                            debugContext("created_input_tensor")),
                        graph()};
  } else if (index == RNNOp::getInputWeightsInIndex()) {
    // Optimized for the forward pass
    // In the forward pass we multiply X * W.transpose()
    return snap::Tensor{poplin::createMatMulInputRHS(
                            graph().getPoplarGraph(),
                            elem_type,
                            {batch_size, input_size},  // LHS
                            {input_size, hidden_size}, // RHS
                            debugContext("created_input_weights_tensor"),
                            {},
                            cache)
                            .transpose()
                            // extending to add the num_directions dimension
                            .expand({0}),
                        graph()};
  } else if (index == RNNOp::getRecurrenceWeightsInIndex()) {
    // Optimized for the forward pass
    // In the forward pass we multiply H * R.transpose()
    return snap::Tensor{poplin::createMatMulInputRHS(
                            graph().getPoplarGraph(),
                            elem_type,
                            {batch_size, hidden_size},  // LHS
                            {hidden_size, hidden_size}, // RHS
                            debugContext("created_recurrence_weights_tensor"),
                            {},
                            cache)
                            .transpose()
                            // extending to add the num_directions dimension
                            .expand({0}),
                        graph()};
  } else if (index == RNNOp::getBiasesInIndex()) {
    return graph().addVariable(elem_type,
                               {num_directions, 2 * hidden_size},
                               poplar::VariableMappingMethod::LINEAR,
                               debugContext("created_bias_tensor"));
  } else if (index == RNNOp::getInitialHInIndex()) {
    // the mapping of this is reset to output[0] in grow
    return graph().addVariable(elem_type,
                               {num_directions, batch_size, hidden_size},
                               poplar::VariableMappingMethod::LINEAR,
                               debugContext("created_initialH_tensor"));
  }

  throw error("RNNOpx::createInput is not supported for index {} of {}",
              index,
              rnn_op.debugName());
}

std::set<TensorId> RNNOpx::mustExistBeforeCreate(InIndex) const { return {}; }

RNNGradOpx::RNNGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<RNNGradOp>(op, Onnx::GradOperators::RNNGrad);
}

void RNNGradOpx::grow(snap::program::Sequence &prog) const {
  auto &rnn_grad_op            = getOp<RNNGradOp>();
  unsigned batch_size          = rnn_grad_op.batch_size;
  unsigned hidden_size         = rnn_grad_op.hidden_size;
  unsigned max_seq_length      = rnn_grad_op.max_seq_length;
  unsigned input_size          = rnn_grad_op.input_size;
  unsigned num_directions      = rnn_grad_op.num_directions;
  auto cache                   = &dv_p->matmulCache;
  auto min_grain_size          = getMinGrainSize();
  const poplar::Type elem_type = getTensorType();
  // fetch forward inputs
  poplar::Tensor forward_input =
      getInTensor(RNNGradOp::getInputInIndex()).getPoplarTensor();
  poplar::Tensor W =
      getInTensor(RNNGradOp::getInputWeightsInIndex()).getPoplarTensor();
  snap::Tensor forward_output =
      getInTensor(RNNGradOp::getFullHiddenStateInIndex());

  // Prepend initialH to forward_output
  snap::Tensor initialH_input;
  if (rnn_grad_op.hasInitialHInput) {
    initialH_input = getInTensor(RNNGradOp::getInitialHInIndex());
  } else {
    initialH_input =
        cloneNcopy(prog,
                   getZerosTensor({num_directions, batch_size, hidden_size},
                                  elem_type,
                                  "zero_initialH_input"));
  }
  forward_output = snap::concat(initialH_input.expand({0}), forward_output, 0);
  if (forward_output.dim(0) > 1)
    graph().getPoplarGraph().setTileMapping(
        forward_output[0].getPoplarTensor(),
        graph().getPoplarGraph().getTileMapping(
            forward_output[1].getPoplarTensor()));

  snap::Tensor full_output_grad = getFullOutputGrad();
  snap::Tensor last_output_grad = getLastOutputGrad();

  // Prepend initialH_grad (which is 0) to full_output_grad
  full_output_grad = snap::concat(
      cloneNcopy(prog,
                 getZerosTensor({1, num_directions, batch_size, hidden_size},
                                elem_type,
                                "initialH_output_grad"),
                 "initialH_output_grad_copy"),
      full_output_grad,
      0);

  // set tile mappings of grads to be the same as the ones for forward_output
  graph().getPoplarGraph().setTileMapping(
      last_output_grad.getPoplarTensor(),
      graph().getPoplarGraph().getTileMapping(
          forward_output[max_seq_length].getPoplarTensor()));
  graph().getPoplarGraph().setTileMapping(
      full_output_grad.getPoplarTensor(),
      graph().getPoplarGraph().getTileMapping(
          forward_output.getPoplarTensor()));

  // dh_prev = full_output_grad[max_seq_length - 1] + last_output_grad
  snap::Tensor dh_prev =
      cloneNcopy(prog, full_output_grad[max_seq_length], "dh_prev")[0];
  snap::popops::addInPlace(graph(),
                           dh_prev,
                           last_output_grad,
                           prog,
                           debugContext("combine_initial_dh"));

  // Set up forward_output_prev to last forward_output value
  snap::Tensor forward_output_prev = cloneNcopy(
      prog, forward_output[max_seq_length], "forward_output_prev")[0];

  // Set up loop index
  poplar::Tensor index =
      getScalarVariable(poplar::UNSIGNED_INT, "bwd_pass_index")
          .getPoplarTensor()
          .reshape({1});
  // i = max_seq_length - 1 (counting downwards to 0)
  popops::fill(graph().getPoplarGraph(),
               index,
               prog.getPoplarSequence(),
               max_seq_length - 1,
               debugContext("initialise_index_to_max_seq_length-1"));

  auto da = snap::Tensor{popops::createSliceableTensor(
                             graph().getPoplarGraph(),
                             elem_type,
                             {max_seq_length, batch_size, hidden_size}, // shape
                             {0},                                       // dims
                             {1},                                       // sizes
                             min_grain_size,
                             debugContext("created_output_tensor")),
                         graph()};
  // Get program for single iteration of the backwards pass
  auto singleStepProg = getBwdStepProg(dh_prev,
                                       full_output_grad,
                                       forward_output_prev,
                                       forward_output,
                                       da,
                                       index);
  // Repeat the program max_seq_length times
  prog.add(snap::program::Repeat(max_seq_length,
                                 singleStepProg,
                                 debugContext("repeat_backward_pass_step")));

  // Perform weight update from accumulated da tensor

  // max_seq_length x hidden_size x input_size,
  // max_seq_length x num_directions x hidden_size x hidden_size
  // -> max_seq_length x hidden_size x (input_size + hidden_size)
  auto X_H = poplar::concat(
      forward_input,
      // squeeze the num_directions dimension, and exclude the last element
      forward_output.squeeze({1}).slice(0, max_seq_length, 0).getPoplarTensor(),
      2);

  // Combine the max_seq_length and batch_size dimensions (0 and 1)
  // Then calculate dW_dR = da.transpose() * X_H
  auto dW_dR =
      poplin::matMul(
          graph().getPoplarGraph(),
          da.flatten(0, 2)
              .transpose()
              .getPoplarTensor(), // hidden_size x (seq_length * batch_size)
          X_H.flatten(
              0, 2), // (seq_length * batch_size) x (input_size + hidden_size)
          prog.getPoplarSequence(),
          debugContext("dW_dR=da*X_H"),
          {},
          cache)
          .reshape({hidden_size, input_size + hidden_size});

  // Retrieve input and recurrence weights grads
  auto dW = snap::Tensor{dW_dR.slice(0, input_size, 1), graph()}.expand({0});
  auto dR = snap::Tensor{dW_dR.slice(input_size, input_size + hidden_size, 1),
                         graph()}
                .expand({0});

  // Combine the max_seq_length and batch_size dimensions (0 and 1) for da
  // Then calculate input grad dX = da * W[0]
  auto dX = snap::Tensor{poplin::matMul(graph().getPoplarGraph(),
                                        da.flatten(0, 2).getPoplarTensor(),
                                        W[0],
                                        prog.getPoplarSequence(),
                                        debugContext("dX"),
                                        {},
                                        cache)
                             .reshape({max_seq_length, batch_size, input_size}),
                         graph()};

  // Set outputs
  setOutTensor(RNNGradOp::getInputOutIndex(), dX);
  setOutTensor(RNNGradOp::getInputWeightsOutIndex(), dW);
  setOutTensor(RNNGradOp::getRecurrenceWeightsOutIndex(), dR);
  if (rnn_grad_op.hasBiasesInput) {
    // Accumulate bias_grad over batch_size index
    snap::Tensor bias_grad_accumulated{
        popops::reduce(graph().getPoplarGraph(),
                       da.getPoplarTensor(),
                       {0, 1},
                       {popops::Operation::ADD},
                       prog.getPoplarSequence(),
                       debugContext("bias_accumulation_over_batch")),
        graph()};
    // Propagate same gradient to both input and hidden bias
    int num_biases              = 2;
    int biases_concat_dimension = 0;
    setOutTensor(
        RNNGradOp::getBiasesOutIndex(),
        bias_grad_accumulated.broadcast(num_biases, biases_concat_dimension)
            .reshape({num_directions, 2 * hidden_size}));
  }
  if (rnn_grad_op.hasInitialHInput) {
    setOutTensor(RNNGradOp::getInitialHOutIndex(), dh_prev.expand({0}));
  }
}

snap::program::Sequence
RNNGradOpx::getBwdStepProg(snap::Tensor &dh_prev,
                           snap::Tensor &full_output_grad,
                           snap::Tensor &forward_output_prev,
                           snap::Tensor &forward_output,
                           snap::Tensor &da,
                           poplar::Tensor &index) const {
  auto &rnn_grad_op            = getOp<RNNGradOp>();
  auto activation              = convert(rnn_grad_op.activation_attribute);
  const poplar::Type elem_type = getTensorType();

  // get recurrence weights tensor
  poplar::Tensor R =
      getInTensor(RNNGradOp::getRecurrenceWeightsInIndex()).getPoplarTensor();

  // For convenience denote:
  //   dL/dh[max_seq_length] = 0
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

  snap::program::Sequence bwdStepProg(debugContext("rnn_bwd_sequence"),
                                      graph());

  // da_next = gradient before applying nonlinearity
  poplar::Tensor da_next =
      popnn::nonLinearityInputGradient(graph().getPoplarGraph(),
                                       activation,
                                       forward_output_prev.getPoplarTensor(),
                                       dh_prev.getPoplarTensor(),
                                       bwdStepProg.getPoplarSequence(),
                                       debugContext("da_next"));
  // Store da[i] = da_next, for weight updates
  popops::dynamicUpdate(graph().getPoplarGraph(),
                        da.getPoplarTensor(),
                        da_next.expand({0}),
                        index,
                        {0},
                        {1},
                        bwdStepProg.getPoplarSequence(),
                        debugContext("update_da"));

  // Fetch the next dh and forward_output values,
  // combining 2 dynamic slices into 1
  auto full_output_grad_dh =
      poplar::concat(full_output_grad.expand({1}).getPoplarTensor(),
                     forward_output.expand({1}).getPoplarTensor(),
                     1);
  auto dh_forward_output_next =
      popops::dynamicSlice(graph().getPoplarGraph(),
                           full_output_grad_dh,
                           index,
                           {0},
                           {1},
                           bwdStepProg.getPoplarSequence(),
                           debugContext("dh_forward_output_next"))[0];
  auto dh_next             = dh_forward_output_next[0];
  auto forward_output_next = snap::Tensor{dh_forward_output_next[1], graph()};

  // dh_recursive = da_next * R[0]
  auto dh_recursive = poplin::matMul(graph().getPoplarGraph(),
                                     da_next,
                                     R[0],
                                     bwdStepProg.getPoplarSequence(),
                                     debugContext("dh_recursive"));

  dh_recursive = popops::rearrange::regroupIfBeneficial(
      graph().getPoplarGraph(),
      dh_recursive,
      dh_next[0],
      bwdStepProg.getPoplarSequence(),
      debugContext("regrouped_dh_recursive"));

  // dh_next += dh_recursive
  popops::addInPlace(graph().getPoplarGraph(),
                     dh_next,
                     dh_recursive,
                     bwdStepProg.getPoplarSequence(),
                     debugContext("dh_next+=dh_recursive"));

  // forward_output_prev = forward_output[i]
  bwdStepProg.add(
      snap::program::Copy(forward_output_next,
                          forward_output_prev,
                          false,
                          debugContext("copy_forward_output_next")));
  // dh_prev = dh_next
  bwdStepProg.add(snap::program::Copy(snap::Tensor{dh_next, graph()},
                                      dh_prev,
                                      false,
                                      debugContext("copy_dh_next")));

  // Decrement index
  popops::subInPlace<unsigned int>(graph().getPoplarGraph(),
                                   index,
                                   1,
                                   bwdStepProg.getPoplarSequence(),
                                   debugContext("increment_bwd_index"));

  return bwdStepProg;
}

std::set<TensorId> RNNGradOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

InputCreatorType RNNGradOpx::getInputCreatorType(InIndex index) const {
  if (index == RNNGradOp::getFullHiddenStateGradInIndex() ||
      index == RNNGradOp::getFullHiddenStateInIndex() ||
      index == RNNGradOp::getInputWeightsInIndex() ||
      index == RNNGradOp::getRecurrenceWeightsInIndex() ||
      index == RNNGradOp::getInputInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

snap::Tensor
RNNGradOpx::createInputTensor(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  auto &rnn_grad_op            = getOp<RNNGradOp>();
  const poplar::Type elem_type = getTensorType();
  unsigned max_seq_length      = rnn_grad_op.max_seq_length;
  unsigned batch_size          = rnn_grad_op.batch_size;
  unsigned input_size          = rnn_grad_op.input_size;
  unsigned hidden_size         = rnn_grad_op.hidden_size;
  unsigned num_directions      = rnn_grad_op.num_directions;
  unsigned min_grain_size      = getMinGrainSize();
  auto cache                   = &dv_p->matmulCache;
  if (index == RNNGradOp::getFullHiddenStateGradInIndex() ||
      index == RNNGradOp::getFullHiddenStateInIndex()) {
    // We want to parallelize over batch_size and hidden_size dimensions
    // We can't parallelize over max_seq_length dimension, so we make sure
    // the tensor can be efficiently sliced along that dimension
    return snap::Tensor{
        popops::createSliceableTensor(
            graph().getPoplarGraph(),
            elem_type,
            {max_seq_length, num_directions, batch_size, hidden_size}, // shape
            {0},                                                       // dims
            {1},                                                       // sizes
            min_grain_size,
            debugContext("created_hidden_state_tensor")),
        graph()};
  } else if (index == RNNGradOp::getInputWeightsInIndex()) {
    // Optimised for the backwards pass
    // In the forward pass we multiply da * W[0]
    return snap::Tensor{poplin::createMatMulInputRHS(
                            graph().getPoplarGraph(),
                            elem_type,
                            {batch_size, hidden_size}, // LHS
                            {hidden_size, input_size}, // RHS
                            debugContext("created_input_weights_tensor"),
                            {},
                            cache)
                            // extending to add the num_directions dimension
                            .expand({0}),
                        graph()};
  } else if (index == RNNGradOp::getRecurrenceWeightsInIndex()) {
    // Optimised for the backwards pass
    // In the forward pass we multiply da * R[0]
    return snap::Tensor{poplin::createMatMulInputRHS(
                            graph().getPoplarGraph(),
                            elem_type,
                            {batch_size, hidden_size},  // LHS
                            {hidden_size, hidden_size}, // RHS
                            debugContext("created_recurrence_weights_tensor"),
                            {},
                            cache)
                            // extending to add the num_directions dimension
                            .expand({0}),
                        graph()};
  } else if (index == RNNGradOp::getInputInIndex()) {
    // We want to parallelize over batch_size and input_size dimensions
    // We can't parallelize over max_seq_length dimension, so we make sure
    // the tensor can be efficiently sliced along that dimension
    return snap::Tensor{popops::createSliceableTensor(
                            graph().getPoplarGraph(),
                            elem_type,
                            {max_seq_length, batch_size, input_size}, // shape
                            {0},                                      // dims
                            {1},                                      // sizes
                            min_grain_size,
                            debugContext("created_input_tensor")),
                        graph()};
  }

  throw error("RNNGradOpx::createInput is not supported for index {} of {}",
              index,
              rnn_grad_op.debugName());
}

poplar::Type RNNGradOpx::getTensorType() const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  return popType(rnn_grad_op.inInfo(RNNGradOp::getInputInIndex()));
}

unsigned RNNGradOpx::getMinGrainSize() const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  return std::max(1,
                  16 / rnn_grad_op.inInfo(RNNGradOp::getInputInIndex())
                           .getDataTypeInfo()
                           ->nbytes());
}

snap::Tensor RNNGradOpx::getLastOutputGrad() const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  if (rnn_grad_op.hasLastHiddenStateGradInput()) {
    return getInTensor(RNNGradOp::getLastHiddenStateGradInIndex());
  } else {
    // Return 0 tensor
    unsigned batch_size          = rnn_grad_op.batch_size;
    unsigned hidden_size         = rnn_grad_op.hidden_size;
    const poplar::Type elem_type = getTensorType();
    auto zero                    = getZerosTensor(
        {1, batch_size, hidden_size}, elem_type, "zero_getLastOutputGrad");
    return zero;
  }
}

snap::Tensor RNNGradOpx::getFullOutputGrad() const {
  auto &rnn_grad_op = getOp<RNNGradOp>();
  if (rnn_grad_op.hasFullHiddenStateGradInput()) {
    auto full_output_grad =
        getInTensor(RNNGradOp::getFullHiddenStateGradInIndex());
    return full_output_grad;
  } else {
    // Return 0 tensor
    const poplar::Type elem_type = getTensorType();
    unsigned batch_size          = rnn_grad_op.batch_size;
    unsigned hidden_size         = rnn_grad_op.hidden_size;
    unsigned max_seq_length      = rnn_grad_op.max_seq_length;

    return getZerosTensor({max_seq_length, 1, batch_size, hidden_size},
                          elem_type,
                          "zero_getFullOutputGrad");
  }
}

namespace {
OpxCreator<RNNOpx> rnnOpxCreator({Onnx::Operators::RNN_7});
OpxCreator<RNNGradOpx> rnnGradOpxCreator(Onnx::GradOperators::RNNGrad);
} // namespace

} // namespace popx
} // namespace popart
