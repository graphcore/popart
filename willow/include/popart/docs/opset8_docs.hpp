// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_DOCS_OPSET8_DOCS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_DOCS_OPSET8_DOCS_HPP_
/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by gen_operators.py.
*/

#define __OPSET8_EXPAND(x) x
#define __OPSET8_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define __OPSET8_VA_SIZE(...)                                                  \
  __OPSET8_EXPAND(__OPSET8_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __OPSET8_CAT1(a, b) a##b
#define __OPSET8_CAT2(a, b) __OPSET8_CAT1(a, b)
#define __OPSET8_DOC1(n1) __doc_##n1
#define __OPSET8_DOC2(n1, n2) __doc_##n1##_##n2
#define __OPSET8_DOC3(n1, n2, n3) __doc_##n1##_##n2##_##n3
#define __OPSET8_DOC4(n1, n2, n3, n4) __doc_##n1##_##n2##_##n3##_##n4
#define __OPSET8_DOC5(n1, n2, n3, n4, n5) __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __OPSET8_DOC6(n1, n2, n3, n4, n5, n6)                                  \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __OPSET8_DOC7(n1, n2, n3, n4, n5, n6, n7)                              \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define OPSET8_DOC(...)                                                        \
  __OPSET8_EXPAND(__OPSET8_EXPAND(__OPSET8_CAT2(                               \
      __OPSET8_DOC, __OPSET8_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
static const char *__doc_popart_abs_opset8 =
    R"doc(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_acos_opset8 =
    R"doc(
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_add_opset8 =
    R"doc(
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_logical_and_opset8 =
    R"doc(
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_argmax_opset8 =
    R"doc(
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.)doc";

static const char *__doc_popart_argmin_opset8 =
    R"doc(
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.)doc";

static const char *__doc_popart_asin_opset8 =
    R"doc(
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_atan_opset8 =
    R"doc(
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_averagepool_opset8 =
    R"doc(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 )doc";

static const char *__doc_popart_batchnormalization_opset8 =
    R"doc(
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    there are multiple cases for the number of outputs, which we list below:

    Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    Output case #2: Y (test mode)
        This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
)doc";

static const char *__doc_popart_cast_opset8 =
    R"doc(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)doc";

static const char *__doc_popart_ceil_opset8 =
    R"doc(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_clip_opset8 =
    R"doc(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)doc";

static const char *__doc_popart_concat_opset8 =
    R"doc(Concatenate a list of tensors into a single tensor)doc";

static const char *__doc_popart_constant_opset8 = R"doc(A constant tensor.)doc";

static const char *__doc_popart_conv_opset8 =
    R"doc(
The convolution operator consumes an input tensor and a filter, and
computes the output.)doc";

static const char *__doc_popart_convtranspose_opset8 =
    R"doc(
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )doc";

static const char *__doc_popart_cos_opset8 =
    R"doc(
Calculates the cosine of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_depthtospace_opset8 =
    R"doc(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
)doc";

static const char *__doc_popart_div_opset8 =
    R"doc(
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_dropout_opset8 =
    R"doc(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
)doc";

static const char *__doc_popart_elu_opset8 =
    R"doc(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)doc";

static const char *__doc_popart_equal_opset8 =
    R"doc(
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_exp_opset8 =
    R"doc(
Calculates the exponential of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_expand_opset8 =
    R"doc(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)doc";

static const char *__doc_popart_flatten_opset8 =
    R"doc(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)doc";

static const char *__doc_popart_floor_opset8 =
    R"doc(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_gru_opset8 =
    R"doc(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
)doc";

static const char *__doc_popart_gather_opset8 =
    R"doc(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
Example 2:
```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [
          [1.0, 1.9],
          [2.3, 3.9],
          [4.5, 5.9],
      ],
  ]
```
)doc";

static const char *__doc_popart_gemm_opset8 =
    R"doc(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).)doc";

static const char *__doc_popart_globalaveragepool_opset8 =
    R"doc(
 GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.)doc";

static const char *__doc_popart_globallppool_opset8 =
    R"doc(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)doc";

static const char *__doc_popart_globalmaxpool_opset8 =
    R"doc(
 GlobalMaxPool consumes an input tensor X and applies max pooling across
 the values in the same channel. This is equivalent to MaxPool with kernel size
 equal to the spatial dimension of input tensor.)doc";

static const char *__doc_popart_greater_opset8 =
    R"doc(
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_hardsigmoid_opset8 =
    R"doc(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)doc";

static const char *__doc_popart_hardmax_opset8 =
    R"doc(
The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the hardmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)doc";

static const char *__doc_popart_identity_opset8 = R"doc(Identity operator)doc";

static const char *__doc_popart_logical_if_opset8 = R"doc(If conditional)doc";

static const char *__doc_popart_instancenormalization_opset8 =
    R"doc(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)doc";

static const char *__doc_popart_lrn_opset8 =
    R"doc(
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
)doc";

static const char *__doc_popart_lstm_opset8 =
    R"doc(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
)doc";

static const char *__doc_popart_leakyrelu_opset8 =
    R"doc(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)doc";

static const char *__doc_popart_less_opset8 =
    R"doc(
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_log_opset8 =
    R"doc(
Calculates the natural log of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_logsoftmax_opset8 =
    R"doc(
The operator computes the logsoftmax (log of softmax) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the logsoftmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)doc";

static const char *__doc_popart_loop_opset8 =
    R"doc(
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]
      %keepgoing[BOOL, scalar]
      %b[INT32, scalar]
    ) {
      %my_local = Add(%a, %b)
      %b_out = Sub(%a, %b)
      %keepgoing_out = Greater(%my_local, %b_out)
      %user_defined_vals = Add(%b, %b)
      return %keepgoing_out, %b_out, %user_defined_vals
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      for (int i=0; i < max_trip_count && keepgoing; ++i) {
        /* User-defined code (loop body) */
        int my_local = a + b; // Reading values in the enclosing scope is fine
        b = a - b; // writes fine if we specify b as a loop-carried dependency
        keepgoing = my_local > b; // keepgoing is a loop-carried dependency
        user_defined_vals[i] = b + b;
        /* End user-defined code */
      }
      // my_local = 123; // Can't do this. my_local was defined in the the body

      // These below values are live-out from the loop and therefore accessible
      b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable a here) are in scope and can
   be referenced in the inputs of the loop.
2) Any variables which you wish to make available in the enclosing scope (i.e.
   the variables b and keepgoing) must be declared as either loop-carried
   dependencies (both at the op inputs and output and at the body net input and
   output) or scan_outputs.
3) Values created in the body cannot be accessed in the enclosing scope.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).
)doc";

static const char *__doc_popart_lpnormalization_opset8 =
    R"doc(
Given a matrix, apply Lp-normalization along the provided axis.
)doc";

static const char *__doc_popart_lppool_opset8 =
    R"doc(
 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)doc";

static const char *__doc_popart_matmul_opset8 =
    R"doc(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)doc";

static const char *__doc_popart_max_opset8 =
    R"doc(
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_maxpool_opset8 =
    R"doc(
 MaxPool consumes an input tensor X and applies max pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 max pooling consisting of computing the max on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is maximum number of elements exclude pad.
 )doc";

static const char *__doc_popart_maxroipool_opset8 =
    R"doc(
 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)doc";

static const char *__doc_popart_mean_opset8 =
    R"doc(
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_min_opset8 =
    R"doc(
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_mul_opset8 =
    R"doc(
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_multinomial_opset8 =
    R"doc(
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
)doc";

static const char *__doc_popart_neg_opset8 =
    R"doc(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_logical_not_opset8 =
    R"doc(
Returns the negation of the input tensor element-wise.
)doc";

static const char *__doc_popart_logical_or_opset8 =
    R"doc(
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_prelu_opset8 =
    R"doc(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).)doc";

static const char *__doc_popart_pad_opset8 =
    R"doc(
Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
)doc";

static const char *__doc_popart_pow_opset8 =
    R"doc(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).)doc";

static const char *__doc_popart_rnn_opset8 =
    R"doc(
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
)doc";

static const char *__doc_popart_randomnormal_opset8 =
    R"doc(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)doc";

static const char *__doc_popart_randomnormallike_opset8 =
    R"doc(
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
)doc";

static const char *__doc_popart_randomuniform_opset8 =
    R"doc(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)doc";

static const char *__doc_popart_randomuniformlike_opset8 =
    R"doc(
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)doc";

static const char *__doc_popart_reciprocal_opset8 =
    R"doc(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_reducel1_opset8 =
    R"doc(
Computes the L1 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducel2_opset8 =
    R"doc(
Computes the L2 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducelogsum_opset8 =
    R"doc(
Computes the log sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducelogsumexp_opset8 =
    R"doc(
Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducemax_opset8 =
    R"doc(
Computes the max of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducemean_opset8 =
    R"doc(
Computes the mean of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducemin_opset8 =
    R"doc(
Computes the min of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reduceprod_opset8 =
    R"doc(
Computes the product of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducesum_opset8 =
    R"doc(
Computes the sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_reducesumsquare_opset8 =
    R"doc(
Computes the sum square of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)doc";

static const char *__doc_popart_relu_opset8 =
    R"doc(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_reshape_opset8 =
    R"doc(
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).)doc";

static const char *__doc_popart_scan_opset8 =
    R"doc(
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops). All these tensors are required to
have the same shape in each iteration of the loop (a restriction imposed to enable efficient
memory allocation). Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs).

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The operation supports batching, and the batch-axis is required to be 0.
When multiple scan_input tensors are used, they must all have the same batch-size,
and they must all have the same maximum-sequence-length (the dimensionality of the
sequence axis or scan axis). The sequence axis or scan axis is required to be 1.

The operation has an optional sequence_lens input (of shape [BATCH_SIZE]) to
allow variable length sequences of length <= the maximum-sequence-length. If this
input is not specified, all sequences are assumed to be of length equal to
maximum-sequence-length. For variable length input sequences, the scan_outputs
will consist of a sequence of same length as the input, padded to the
maximum-sequence-length.

The optional attribute directions can be used to scan a sequence in the reverse direction.
If this attribute is omitted, all sequences are scanned in the forward direction.
A bidirectional scan be performed by specifying the same tensor input twice in the
scan_inputs, once with a forward direction, and once with a backward direction.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body
    > (sequence_lengths, init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // T.shape[0] denotes the batch-size of T
    // The batch-size of scan_1, ..., scan_m are all required to be equal
    batch_size = scan_1.shape[0];

    // scan_i.shape[1] denotes the (max) sequence-length of scan_i
    // scan_i.shape[1] is required to be equal to scan_j.shape[1] for all i,j.
    max_sequence_length = scan_1.shape[1];

    for (int batch = 0; batch < batch_size; ++batch) {
        // initialize state-variables
        st_1 = init_1; ... st_n = init_n;
        // initialize scan-output variables: [] denotes an empty tensor
        scan_out_1 = []; ...; scan_out_k = [];
        // identify number of iterations:
        N = (sequence_lengths specified) ? sequence_lengths[batch] : max_sequence_length;

        // execute loop
        for (int t = 0; t < N; ++t) {
            // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
            // of rank one less than T obtained by indexing T at position t along axis k.
            si_1 = (scan_1<axis=0>[batch])<axis=1>[t];
            ... ;
            si_m = (scan_m<axis=0>[batch])<axis=1>[t];
            // execute loop-body
            st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
            // accumulate the scan-output elements
            scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
        }
        // accumulate the outputs for this batch:
        bst_1[batch] = st_1; ..., bst_n[batch] = st_n;
        // Note scan-outputs will have size max_sequence_length, but only first N values will be meaningful.
        // The remaining values have an undefined value.
        b_scan_out_1[batch] = scan_out_1; ...; b_scan_out_k[batch] = scan_out_k;
    }
    return bst_1, ..., bst_n, b_scan_out_1, ..., b_scan_out_k;



*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1]("", %H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }

)doc";

static const char *__doc_popart_selu_opset8 =
    R"doc(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)doc";

static const char *__doc_popart_shape_opset8 =
    R"doc(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
)doc";

static const char *__doc_popart_sigmoid_opset8 =
    R"doc(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)doc";

static const char *__doc_popart_sin_opset8 =
    R"doc(
Calculates the sine of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_size_opset8 =
    R"doc(
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
)doc";

static const char *__doc_popart_slice_opset8 =
    R"doc(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)doc";

static const char *__doc_popart_softmax_opset8 =
    R"doc(
The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)doc";

static const char *__doc_popart_softplus_opset8 =
    R"doc(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)doc";

static const char *__doc_popart_softsign_opset8 =
    R"doc(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)doc";

static const char *__doc_popart_spacetodepth_opset8 =
    R"doc(SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
)doc";

static const char *__doc_popart_split_opset8 =
    R"doc(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)doc";

static const char *__doc_popart_sqrt_opset8 =
    R"doc(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)doc";

static const char *__doc_popart_squeeze_opset8 =
    R"doc(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)doc";

static const char *__doc_popart_sub_opset8 =
    R"doc(
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_sum_opset8 =
    R"doc(
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

static const char *__doc_popart_tan_opset8 =
    R"doc(
Calculates the tangent of the given input tensor, element-wise.
)doc";

static const char *__doc_popart_tanh_opset8 =
    R"doc(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)doc";

static const char *__doc_popart_tile_opset8 =
    R"doc(Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
)doc";

static const char *__doc_popart_topk_opset8 =
    R"doc(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)doc";

static const char *__doc_popart_transpose_opset8 =
    R"doc(
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)doc";

static const char *__doc_popart_unsqueeze_opset8 =
    R"doc(
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
)doc";

static const char *__doc_popart_upsample_opset8 =
    R"doc(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)doc";

static const char *__doc_popart_logical_xor_opset8 =
    R"doc(
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)doc";

#endif // POPART_WILLOW_INCLUDE_POPART_DOCS_OPSET8_DOCS_HPP_
