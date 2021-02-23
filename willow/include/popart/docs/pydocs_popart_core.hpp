/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x) x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define __VA_SIZE(...) __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b) a##b
#define __CAT2(a, b) __CAT1(a, b)
#define __DOC1(n1) __doc_##n1
#define __DOC2(n1, n2) __doc_##n1##_##n2
#define __DOC3(n1, n2, n3) __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4) __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5) __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                                         \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)                                     \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                                               \
  __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static const char *__doc_popart_AccumulateOuterFragmentSchedule =
    R"doc(Enum type that determines how the operations in the accumulate outer
fragment will be scheduled accross virtual graphs (only relevant to
pipelined modes).)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapCycleOptimized =
        R"doc(Try and parallelise ops with different virtual graph IDs as much as
possible.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapMemoryOptimized =
        R"doc(Try and parallelise ops with different virtual graph IDs but avoid
certain steps that are costly in terms of memory usage.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Scheduler =
    R"doc(Don't add additional constraints and let the scheduler work it out.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Serial =
    R"doc(Add constraints that ensure ops are executed in virtual graph ID
order.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings =
    R"doc(A structure containing accumulate outer fragment settings.)doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings =
        R"doc()doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings_2 =
        R"doc()doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_excludedVirtualGraphs =
        R"doc(A setting to explicitly tell PopART to avoid to try and parallelise
the given virtual graph ids. This setting is experimental and may
change.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_operator_assign = R"doc()doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings_schedule =
    R"doc(Tell PopART how you would like to schedule the accumulate outer
fragment. This setting is experimental and may change.)doc";

static const char *__doc_popart_Adam =
    R"doc(AdamW, Lamb and AdaMax optimizer implementation.

Akin to any optimizer implementation, this class is responsible for
updating each weight tensor ($w$) in the model using the gradient
($g$) of the loss function with respect to the weight as calculated
during the backwards pass.

The optimizer has the following **state** for each weight:

* *first-order momentum* ($m$) * *second-order momentum* ($v$) * *time
step* ($t$)

The optimizer has the following **hyper parameters**:

* *learning rate* ($\text{lr}$) * *weight decay* ($\text{wd}$) *
*beta1* ($\beta_1$) * *beta2* ($\beta_2$) * *epsilon* ($\epsilon$) *
*loss scaling* ($\text{ls}$) * *maximum weight norm* ($\text{mwn}$)

The values of these parameters can be shared between all weights but
some can be overridden with weight-specific values (see
Adam::insertSpecific). Hyper parameters are captured using
OptimizerValue objects and therefore can be either a constant value or
a non-constant value that can be adjusted by the user.

The values of #AdamMode and #WeightDecayMode passed to the constructor
determines how weights are updated (see below).

In the following we will describe how this optimizer updates a weight
using a gradient. In the context of this description the gradient is
is the value of the gradient *after* any gradient accumulation has
been performed and *after* the application of a loss scaling factor to
the gradient has been corrected for.

When the optimizer needs to update a weight, $w$, using a gradient,
$g$, it first computes a term $g_\text{tmp}$, which is effectively is
$g$ with L2 regularization applied if the #WeightDecayMode is set to
WeightDecayMode::L2Regularization this, as follows:

\f[ g_\text{tmp} := \left\{\begin{aligned} g & \text{ \; (Decay) } \\
(g + \text{wd} * w) & \text{ \; (L2Regularization) \; . } \\
\end{aligned}\right.\\ \f]

Secondly, the optimizer updates the optimizer state as follows:

\f[ m' &:= \beta_1 * m + (1 - \beta_1) * g_\text{tmp} \\ v' &:=
\left\{\begin{aligned} \beta_2 * v + (1 - \beta_2) * g_\text{tmp}^2 &
\text{ \; (Adam/AdamNoBias) } \\ \beta_2 * v + (1 - \beta_2) *
g_\text{tmp}^2 & \text{ \; (Lamb/LambNoBias) } \\ \text{max}(\beta_2 *
v, |g_\text{tmp}|) & \text{ \; (AdaMax) } \\ \end{aligned}\right.\\ t'
&:= t + 1 \\ \f]

Next, it computes the following terms:

\f[ m_\text{tmp} &:= \left\{\begin{aligned} m' & \text{ \;
(AdamNoBias/LambNoBias) } \\ \frac{m'}{(1 - \beta_1^{t'})} & \text{ \;
(Adam/Lamb/AdaMax) } \\ \end{aligned}\right.\\ v_\text{tmp} &:=
\left\{\begin{aligned} v' & \text{ \; (AdamNoBias/LambNoBias) } \\
\frac{v'}{(1 - \beta_2^{t'})} & \text{ \; (Adam/Lamb/AdaMax) } \\
\end{aligned}\right.\\ u_\text{tmp} &:= \left\{\begin{aligned}
\frac{m_\text{tmp}}{(\sqrt{v_\text{tmp}} + \epsilon)} + \text{wd} * w
&\text{ \; (Decay) } \\ \frac{m_\text{tmp}}{(\sqrt{v_\text{tmp}} +
\epsilon)} &\text{ \; (L2Regularization) } \\ \end{aligned}\right. \f]

Finally, the optimizer updates the weight as follows:

\f[ w' := \left\{\begin{aligned} w - \text{lr} * u_\text{tmp} &\text{
\; (Adam/AdamNoBias/AdaMax) } \\ w -
\biggl(\frac{\text{min}(\lVert{w}\rVert,
\text{mwn})}{\lVert{u_\text{tmp}}\rVert}\biggr) * \text{lr} *
u_\text{tmp} &\text{ \; (Lamb/LambNoBias) } \\ \end{aligned}\right.
\f]

In addition to the above, the *loss scaling* hyper parameter is
similar in nature to the velocity scaling parameter. It is a scaling
value that is applied to the loss gradient at the start of the the
backwards pass and, at the end of the backwards pass, this scaling is
reversed by multiplying the gradients for each weight with the inverse
of the loss scaling value prior to updating the optimizer state. Using
loss scaling can also improve numerical stability in some cases.

**NOTE**: The maximum weight norm is referred to as $\phi$ in [You et
al., 2020](https://arxiv.org/abs/1904.00962).)doc";

static const char *__doc_popart_AdamMode =
    R"doc(Enum type describing the mode of an Adam optimizer instance.)doc";

static const char *__doc_popart_AdamMode_AdaMax = R"doc(Adamax mode.)doc";

static const char *__doc_popart_AdamMode_Adam =
    R"doc(Adam or AdamW mode, depending on weight decay setting (see [Kingma &
Ba, 2015](https://arxiv.org/abs/1412.6980) and [Loshchilov & Hutter,
2018](https://arxiv.org/pdf/1711.05101.pdf)).)doc";

static const char *__doc_popart_AdamMode_AdamNoBias =
    R"doc(Like Adam but without bias correction.)doc";

static const char *__doc_popart_AdamMode_Lamb =
    R"doc(Lamb mode (see [You et al., 2020](https://arxiv.org/abs/1904.00962)).)doc";

static const char *__doc_popart_AdamMode_LambNoBias =
    R"doc(Like Lamb but without bias correction.)doc";

static const char *__doc_popart_Adam_Adam =
    R"doc(Constructor.

Parameter ``defaultLearningRate``:
    the learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    the weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultBeta1``:
    the beta1 value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultBeta2``:
    the beta2 value value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultEps``:
    the epsilon value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``lossScaling``:
    the loss scaling value to use.

Parameter ``maxWeightNorm``:
    the maxWeightNorm value to use.

Parameter ``adamMode``:
    the AdamMode value to use.

Parameter ``weightDecayMode``:
    the WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    the maxWeightNorm value to use.

Parameter ``accumType``:
    data type to use for gradient accumulation.

Parameter ``accl1Type``:
    data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    data type to use for tensor that stores second-order momentum
    optimizer state.)doc";

static const char *__doc_popart_Adam_Adam_2 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_3 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_4 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_5 =
    R"doc(Constructor.

Parameter ``params``:
    a parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"`, and the
    map's values pairs of floats and booleans representing
    OptimizerValue constructor arguments. The map does not have to
    specify each hyper parameter as default values will be used where
    parameters are missing.

Parameter ``adamMode``:
    the AdamMode value to use.

Parameter ``weightDecayMode``:
    the WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    the maxWeightNorm value to use.

Parameter ``accumType``:
    data type to use for gradient accumulation.

Parameter ``accl1Type``:
    data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    data type to use for tensor that stores second-order momentum
    optimizer state.

**EXAMPLE**:

```
Adam({{"defaultLearningRate", {0.02, False}},
      {"defaultBeta1", {0.9, True}},
      {"defaultBeta2":{0.999, True}}},
      AdamMode::Adam,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT);
```)doc";

static const char *__doc_popart_Adam_Adam_6 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_7 = R"doc()doc";

static const char *__doc_popart_Adam_accl1Type = R"doc()doc";

static const char *__doc_popart_Adam_accl2Type = R"doc()doc";

static const char *__doc_popart_Adam_accumType = R"doc()doc";

static const char *__doc_popart_Adam_b1helper = R"doc()doc";

static const char *__doc_popart_Adam_b1s = R"doc()doc";

static const char *__doc_popart_Adam_b2helper = R"doc()doc";

static const char *__doc_popart_Adam_b2s = R"doc()doc";

static const char *__doc_popart_Adam_beta1s = R"doc()doc";

static const char *__doc_popart_Adam_beta2s = R"doc()doc";

static const char *__doc_popart_Adam_clone = R"doc()doc";

static const char *__doc_popart_Adam_createOp = R"doc()doc";

static const char *__doc_popart_Adam_decayMode = R"doc()doc";

static const char *__doc_popart_Adam_epshelper = R"doc()doc";

static const char *__doc_popart_Adam_epss = R"doc()doc";

static const char *__doc_popart_Adam_epsvs = R"doc()doc";

static const char *__doc_popart_Adam_fromDefaultMap = R"doc()doc";

static const char *__doc_popart_Adam_getComplete = R"doc()doc";

static const char *__doc_popart_Adam_getInputIds = R"doc()doc";

static const char *__doc_popart_Adam_getOptimizerInputs = R"doc()doc";

static const char *__doc_popart_Adam_getStoredValue = R"doc()doc";

static const char *__doc_popart_Adam_getUnsetBeta1 =
    R"doc(Default beta1 value.)doc";

static const char *__doc_popart_Adam_getUnsetBeta2 =
    R"doc(Default beta2 value.)doc";

static const char *__doc_popart_Adam_getUnsetEps =
    R"doc(Default epsilon value.)doc";

static const char *__doc_popart_Adam_getUnsetLearningRate =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_Adam_getUnsetLossScaling =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_Adam_getUnsetMaxWeightNorm =
    R"doc(Default maximum weight norm value.)doc";

static const char *__doc_popart_Adam_getUnsetWeightDecay =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_Adam_gshelper = R"doc()doc";

static const char *__doc_popart_Adam_hasSpecific = R"doc()doc";

static const char *__doc_popart_Adam_hash = R"doc()doc";

static const char *__doc_popart_Adam_insertSpecific =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    the TensorId of the weight.

Parameter ``learningRate``:
    the learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    the weight decay value to use for this specific weight.

Parameter ``beta1``:
    the beta1 value to use for this specific weight.

Parameter ``beta2``:
    the beta2 value to use for this specific weight.

Parameter ``eps``:
    the epsilon value to use for this specific weight.)doc";

static const char *__doc_popart_Adam_insertSpecific_2 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    the TensorId of the weight.

Parameter ``params``:
    a parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"` and the map's
    values pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_Adam_learningRates = R"doc()doc";

static const char *__doc_popart_Adam_lrhelper = R"doc()doc";

static const char *__doc_popart_Adam_lrs = R"doc()doc";

static const char *__doc_popart_Adam_lshelper = R"doc()doc";

static const char *__doc_popart_Adam_maxWeightNorms = R"doc()doc";

static const char *__doc_popart_Adam_mode = R"doc()doc";

static const char *__doc_popart_Adam_mwnhelper = R"doc()doc";

static const char *__doc_popart_Adam_mwns = R"doc()doc";

static const char *__doc_popart_Adam_resetTensorData = R"doc()doc";

static const char *__doc_popart_Adam_runValueChecks = R"doc()doc";

static const char *__doc_popart_Adam_setStep = R"doc()doc";

static const char *__doc_popart_Adam_setStep_2 = R"doc()doc";

static const char *__doc_popart_Adam_setStep_3 = R"doc()doc";

static const char *__doc_popart_Adam_setTensorData = R"doc()doc";

static const char *__doc_popart_Adam_type = R"doc()doc";

static const char *__doc_popart_Adam_type_s = R"doc()doc";

static const char *__doc_popart_Adam_validReplacement = R"doc()doc";

static const char *__doc_popart_Adam_wdhelper = R"doc()doc";

static const char *__doc_popart_Adam_wds = R"doc()doc";

static const char *__doc_popart_Adam_weightDecays = R"doc()doc";

static const char *__doc_popart_AddPatternName = R"doc()doc";

static const char *__doc_popart_AddPatternName_AddPatternName = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_AiGraphcoreOpset1 =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_atan2 =
    R"doc(Add an atan2 operation to the model

Returns the element-wise angle theta as a tensor, -pi < theta <= pi,
such that for two input tensors x and y and given r != 0, x = r cos
theta, and y = r sin theta, element-wise.

In the case of x > 0, theta = arctan(y/x)

Parameter ``args``:
    A vector of inputs tensors (y, x)

Parameter ``name``:
    Optional identifier for operation

Returns:
    The name of the result tensor containing element wise theta values)doc";

static const char *__doc_popart_AiGraphcoreOpset1_call =
    R"doc(Add a call operation to the model

This is a poplar extension, to expose manual code re-use to the
builder

Parameter ``args``:
    Tensor T

Parameter ``callee``:
    The subgraph to call into

Parameter ``debugContext``:
    Optional debug context

Returns:
    A vector of tensors; the subgraph outputs)doc";

static const char *__doc_popart_AiGraphcoreOpset1_depthtospace =
    R"doc(Add the 'DepthToSpace' to the model (This allows DepthToSpace_11 to be
targeted from earlier opsets)

The purpose of Depth to Space, also known as pixel shuffling, is to
rearrange data from the depth (channels) dimension into the spacial
(width and height) dimensions. It is an efficient means of learning
upsampling alongside mixing convolution with bilinear interpolation
and using transpose convolution.

https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpac
e

Parameter ``args``:
    List containing single tensor input

Parameter ``blocksize``:
    Indicates the scale factor: if the input is [N, C, H, W] and the
    blocksize is B, the output will be [N, C/(B*B), H*B, W*B]

Parameter ``mode``:
    Specifies how the data is rearranged "DCR": depth-column-row order
    "CRD": column-row-depth order

Parameter ``debugContext``:
    Optional debug context

Returns:
    A tensor which is a rearrangement of the input tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_detach =
    R"doc(Add a detach operation to the model

Parameter ``args``:
    Tensor T Input tensor.

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicadd =
    R"doc(Add a dynamic add operation to the model

Creates a copy of "tensor" with "slice" added at "offset", e.g. out =
tensor, out[offset] += slice

Parameter ``args``:
    [tensor, offset, slice]

Parameter ``axes``:
    Axes along which to add

Parameter ``sizes``:
    Size of the slice in each axis

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicslice =
    R"doc(Add a dynamic slice operation to the model

Creates a new slice tensor, e.g. slice = tensor[offset]

Parameter ``args``:
    [tensor, offset]

Parameter ``axes``:
    Axes along which to slice

Parameter ``sizes``:
    Size of the slice in each axis

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicupdate =
    R"doc(Add a dynamic update operation to the model

Creates a copy of "tensor" with "slice" inserted at "offset", e.g. out
= tensor, out[offset] = slice

Parameter ``args``:
    [tensor, offset, slice]

Parameter ``axes``:
    Axes along which to update

Parameter ``sizes``:
    Size of the slice in each axis

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamiczero =
    R"doc(Add a dynamic zero operation to the model

Creates a copy of "tensor" with a slice at "offset" set to zero, e.g.
out = tensor, out[offset] = 0.0

Parameter ``args``:
    [tensor, offset]

Parameter ``axes``:
    Axes along which to erase

Parameter ``sizes``:
    Size of the slice in each axis

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_expm1 =
    R"doc(Add expm1 operation to the model It computes exp(x) - 1. Calculates
the exponential of the given input tensor and subtract one. Element-
wise.

Parameter ``args``:
    Tensor T

Parameter ``name``:
    Optional identifier for operation

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_fmod =
    R"doc(Add fmod operation to the model.

This is equivalent to C's fmod function. The result has the same sign
as the dividend.

Parameter ``args``:
    Input tensors.

Returns:
    Computes the element-wise remainder of division. The remainder has
    the same sign as the dividend.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_gelu =
    R"doc(Add a gelu operation to the model

This is a poplar extension, to replace the experimental scale operator
that has been removed

Parameter ``args``:
    Tensor T

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_groupnormalization =
    R"doc(Add a groupnormalization operation to the model

This is a poplar extension

The group will be created from a strided input

Parameter ``args``:
    A vector of input tensors (x, scale, bias)

Parameter ``num_groups``:
    The number of groups to separate the channels into

Parameter ``epsilon``:
    The epsilon value to use to avoid division by zero.

Parameter ``debugContext``:
    Optional debug context

Returns:
    A vector of tensors (y, mean, var))doc";

static const char *__doc_popart_AiGraphcoreOpset1_identityloss =
    R"doc(Add an identity loss operation to the model

Calculates the loss using the identity operator.

Parameter ``args``:
    [tensor]

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``name``:
    Optional identifier for operation

Parameter ``debugContext``:
    Optional debug context)doc";

static const char *__doc_popart_AiGraphcoreOpset1_ctcloss =
    R"doc(Add an connectionist temporal classification (CTC) loss operation to
the model.

With T being maximum input length, N being batch size, C being number of
classes, S being a maximum target length, this op calculates the CTC loss
for a logarithmised probabilities tensor with shape [T, N, C], a class
target tensor with shape [N, S], a input lengths tensor [N] and a target
lengths tensor [N].

Note that C includes a blank class (default=0). The probabilities tensor
is padded as required. Target sequences are also padded and are
populated with values <=C not including the blank class, up to their
respective target lengths. Note that target lengths cannot exceed input
lengths.

Parameter ``args``:
    [log_probs,targets,input_lengths,target_lengths]

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``blank``:
    The integer representing the blank class.

Parameter ``debugContext``:
    Optional debug context)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init =
    R"doc(Add an init operation to the model

Parameter ``shape``:
    Shape of the tensor to initialise

Parameter ``data_type``:
    Data type to initialise tensor with

Parameter ``init_type``:
    Mode of tensor initialisations

Parameter ``batch_axis``:
    Axis relative to batch size

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init_2 =
    R"doc(Add an init operation to the model

Parameter ``shape``:
    Shape of the tensor to initialise

Parameter ``data_type``:
    Data type to initialise tensor with

Parameter ``init_type``:
    Mode of tensor initialisations

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_l1loss =
    R"doc(Add an l1 loss operation to the model

Calculates the mean absolute error between each element in the input
with a zero target

Parameter ``args``:
    [tensor]

Parameter ``lambda``:
    Scale factor of L1 loss

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_log1p =
    R"doc(Add log1p operation to the model It computes log(x + 1). Calculates
the logarithm of the given input tensor plus one. Element-wise.

Parameter ``args``:
    Tensor T

Parameter ``name``:
    Optional identifier for operation

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_lstm = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_multiconv =
    R"doc(Add a multi-convolution to the model

Using this multi-convolution API ensures that the convolutions are
executed in parallel on the device,

Functionally, a multi-convolution is equivalent to a series of single
convolutions. Using this multi-convolution API is always equivalent to
calling the single-convolution API (conv) once for each argument.

For example, calling

> A0 = conv({X0, W0, B0}) > A1 = conv({X1, W1})

is functionally equivalent to calling

> {A0, A1} = multiconv({{X0, W0, B0}, {X1, Q1}).

It is possible that any two convolutions cannot be executed in
parallel due to topological constraints. For example,

> B = conv({A, W0}); > C = B + A > D = conv({C, W1});

cannot be converted to,

> {B, D} = multiconv({{A, W0}, {C, W1}}).

Note that it is not possible to create such a cycle by adding a multi-
convolution with this API.

Calls to multiconv eventually map to PopLibs'
poplin::multiconv::convolution

Parameter ``tensors``:
    List of {DataId, WeightId, BiasId (optional)}

Parameter ``dilations``:
    The dilations attributes for each convolution.

Parameter ``pads``:
    The pads for each convolution.

Parameter ``strides``:
    The strides for each convolution.

Parameter ``availableMemoryProportions``:
    The available memory proportions per conv, each [0, 1).

Parameter ``partialsTypes``:
    The partials type per conv

Parameter ``planType``:
    Run convolutions in parallel or series.

Parameter ``perConvReservedTiles``:
    Tiles to reserve per convolution when planning.

Parameter ``cycleBackOff``:
    Cycle back off proportion, [0, 1).

Parameter ``debugContext``:
    Optional debug context

All input vectors must be either empty, or equal in length to the
number of convolutions. Note that groups for each convolution are
automatically inferred from the shapes of the data and weight inputs.

Returns:
    The TensorId of the output Tensor from each convolution.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nllloss =
    R"doc(Add a negative log-likelihood loss operation to the model

Calculates the nll loss given a probability tensor over classes, and a
target tensor containing class labels

Parameter ``args``:
    [probs, target]

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``ignoreIndex``:
    Optional class index to ignore in loss calculation

Parameter ``inputIsLogProbability``:
    Specifies if the input tensor contains log-probabilities or raw
    probabilities (false, default).

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nop =
    R"doc(Add a nop operation to the model

Parameter ``args``:
    Tensor T

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_printtensor =
    R"doc(Add a print tensor operation to the model

This is a poplar extension)doc";

static const char *__doc_popart_AiGraphcoreOpset1_remainder = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_replicatedallreduce =
    R"doc(Add a replicated all reduce operation to the model

This is a poplar extension, to expose manual code re-use to the
builder

Parameter ``args``:
    Tensor T to reduce across

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reshape =
    R"doc(Add reshape operation to the model. Reshape the input tensor. This
reshape takes the shape to reshape into as an attribute instead of a
tensor input as the onnx reshape op.

Parameter ``arg``:
    Single input Tensor.

Parameter ``shape``:
    The shape of the output Tensor. The output Tensor must contain the
    same number of elements as the input Tensor.

Parameter ``name``:
    Optional identifier for operation

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_round =
    R"doc(Add the 'Round' to the model (This allows Round_11 to be targeted from
earlier opsets)

https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round

Parameter ``args``:
    List of input tensor ids

Parameter ``debugContext``:
    Optional debug context

Returns:
    The normalized output tensor ids)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scale =
    R"doc(Add a scale operation to the model

This is a poplar extension, to replace the experimental scale operator
that has been removed

Parameter ``args``:
    Tensor T

Parameter ``scale``:
    The scale to apply

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scaledadd =
    R"doc(Add a scaledadd operation to the model X = scale0 * T0 + scale1 * T1

Parameter ``args``:
    Tensor {T0, T1, scale0, scale1}

Parameter ``scale0``:
    The scale to apply (if no scale0 tensor is supplied)

Parameter ``scale1``:
    The scale to apply (if no scale1 tensor is supplied)

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_shapeddropout =
    R"doc(Add a shaped dropout operation to the model

Applies a shaped dropout to the input tensor. This operator requires a
shape parameter that is used to define the shape of the dropout mask
so that strongly correlated features in the input tensor can be
preserved. The provided shape must be broadcastable to the input
tensor. Note that this operation targets the poprand library function
of the same name.

Parameter ``args``:
    [tensor]

Parameter ``shape``:
    Shape of dropout mask. Must be broadcastable to the input.

Parameter ``ratio``:
    Probability of dropping an input feature (default = 0.5).

Parameter ``name``:
    Optional identifier for operation

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_subsample =
    R"doc(Add a subsample operation to the model

This is a poplar extension

If multiple tensors are provided that strides will applied to them all

Parameter ``args``:
    Tensor T

Parameter ``strides``:
    The strides

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiOnnxMlOpset1 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_AiOnnxMlOpset1 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_AnchorReturnType =
    R"doc(A class that captures an AnchorReturnTypeId value and, when this value
is AnchorReturnTypeId::EVERYN, the associated `N` number. The
constructor takes `std::string` values and converts them as
appropriate.

See also: #AnchorReturnTypeId.)doc";

static const char *__doc_popart_AnchorReturnTypeId =
    R"doc(An anchor tensor is a tensor that the user wants returned after a call
to Session::run. Each call to Session::run results in `batchesPerStep
x accumulationFactor x replicationFactor` of such tensors being
computed. We refer to the samples associated with each such
computation as a micro batch. The dimensions are user-specified by the
following parameters:

* `batchesPerStep` is the value in DataFlow. * `accumulationFactor` is
the value defined by SessionOptions::accumulationFactor. *
`replicationFactor` is the value defined by
SessionOptions::globalReplicationFactor.

This enum type describes the strategy with which the micro batch
values for anchor tensors (or summaries thereof) are written or to the
IStepIO instance passed to Session::run.

See also: AnchorReturnType.

**NOTE**: Anchors are essentially what tensorflow calls "fetches".)doc";

static const char *__doc_popart_AnchorReturnTypeId_All =
    R"doc(Return the tensor value for *all* micro batches for each replica.

The buffer shape required for this anchor in IStepIO is
`[batchesPerStep, accumulationFactor, replicationFactor,
<anchorTensorShape>]` (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_EveryN =
    R"doc(Return the tensor value for every `N`th global batch for each replica
and for all accumulation steps in that global batch. Note that the
value of `N` is captured by AnchorReturnType.

The buffer shape required for this anchor in IStepIO is
`[batchesPerStep // N, accumulationFactor, replicationFactor,
<anchorTensorShape>]` (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Final =
    R"doc(Only return the tensor value for the last micro batch of the
Session::run call for each replica.

The buffer shape required for this anchor in IStepIO is
`[replicationFactor, <anchorTensorShape>]` (with dimensions of size 1
removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Sum =
    R"doc(Return one tensor value for each replica, doing a sum reduction over
the `batchesPerStep` and `accumulationFactor` dimensions.

The buffer shape required for this anchor in IStepIO is
`[replicationFactor, <anchorTensorShape>]` (with dimensions of size 1
removed).)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType =
    R"doc(Constructor.

Parameter ``artString``:
    - the string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * `"final"` -
    AnchorReturnTypeId::FINAL * `"all"` - AnchorReturnTypeId::ALL *
    `"sum"` - AnchorReturnTypeId::SUM

**NOTE**: Constructing an AnchorReturnType with of type
AnchorReturnTypeId::EVERYN using this constructor will result in an
error. Use the constructor that also specifies a return period.)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType_2 =
    R"doc(Constructor.

Parameter ``artString``:
    the string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * `"final"` -
    AnchorReturnTypeId::FINAL * `"everyn"` -
    AnchorReturnTypeId::EVERYN * `"all"` - AnchorReturnTypeId::ALL *
    `"sum"` - AnchorReturnTypeId::SUM

Parameter ``returnPeriod``:
    the value of `N` in case of AnchorReturnTypeId::EVERYN.

**NOTE**: Constructing a #AnchorReturnType with of type
AnchorReturnTypeId::EVERYN will result in an error. Use the
constructor that also specifies the return period.)doc";

static const char *__doc_popart_AnchorReturnType_artId = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_artStr = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_getIdFromStr = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_hash = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_id = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_returnPeriod = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_rp = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_str = R"doc()doc";

static const char *__doc_popart_BatchSerializationBatchSchedule =
    R"doc(Enum type that describes how to change the batch serialisation
subgraph schedule before outlining. **NOTE:** This setting is
experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Isomorphic =
    R"doc(Encourage all ops within batch subgraphs to be scheduled identically
and for each subgraph to be scheduled in sequence (good for
outlineability).)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_N =
    R"doc(The number of BatchSerializationBatchSchedule values.)doc";

static const char
    *__doc_popart_BatchSerializationBatchSchedule_OverlapOnCompute =
        R"doc(OverlapOnCompute tries to put the RemoteLoad for batch N+1 right
before the compute phase of batch N.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_OverlapOnIo =
    R"doc(OverlapOnIo tries to put the RemoteLoad for batch N+1 right after the
compute phase of batch N.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Scheduler =
    R"doc(Don't encourage any particular scheduling for ops within batch
subgraphs (leave it to the scheduler) but tell the scheduler to
schedule subgraphs in sequence.)doc";

static const char *__doc_popart_BatchSerializationMethod =
    R"doc(Enum type that describes how to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationMethod_Loop =
    R"doc(Loop over the batch dimension)doc";

static const char *__doc_popart_BatchSerializationMethod_N =
    R"doc(The number of BatchSerializationMethod values.)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollDynamic =
    R"doc(Unroll the batch with dynamic slicing)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollStatic =
    R"doc(Unroll the batch with static slicing)doc";

static const char *__doc_popart_BatchSerializationSettings =
    R"doc(A structure containing batch serialization settings.)doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings =
        R"doc()doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings_2 =
        R"doc()doc";

static const char *__doc_popart_BatchSerializationSettings_batchSchedule =
    R"doc(Experimental value that changes how operations are scheduled.)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnExecutionPhaseChange =
        R"doc(Break batch serialization chains when the execution phase changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnPipelineStageChange =
        R"doc(Break batch serialization chains when the pipeline stage changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnVirtualGraphChange =
        R"doc(Break batch serialization chains when the virtual graph changes (by
concatenating the compute batches to the local batch).)doc";

static const char *__doc_popart_BatchSerializationSettings_factor =
    R"doc(The number of compute batches to split operations into.)doc";

static const char *__doc_popart_BatchSerializationSettings_method =
    R"doc(Experimental value to control how batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationSettings_operator_assign =
    R"doc()doc";

static const char *__doc_popart_BatchSerializationSettings_transformContext =
    R"doc(Experimental value to control when batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationTransformContext =
    R"doc(Enum type that describes when to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Bwd =
    R"doc(Apply after growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Fwd =
    R"doc(Apply before growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_N =
    R"doc(The number of BatchSerializationTransformContext values.)doc";

static const char *__doc_popart_Builder =
    R"doc(An interface for a Builder, used for creating ONNX graphs.)doc";

static const char *__doc_popart_Builder_2 =
    R"doc(An interface for a Builder, used for creating ONNX graphs.)doc";

static const char *__doc_popart_Builder_3 = R"doc()doc";

static const char *__doc_popart_BuilderImpl = R"doc()doc";

static const char *__doc_popart_Builder_Builder = R"doc()doc";

static const char *__doc_popart_Builder_addInitializedInputTensor =
    R"doc(Add a new preinitialized input tensor to the model

Parameter ``initData``:
    The initial data of the input tensor

Parameter ``debugContext``:
    Optional debug information

Returns:
    The unique name of the input tensor)doc";

static const char *__doc_popart_Builder_addInputTensor =
    R"doc(Add a new input tensor to the model

Parameter ``tensorInfo``:
    The shape and type of the input tensor

Parameter ``debugContext``:
    Optional debug information

Returns:
    The unique name of the input tensor)doc";

static const char *__doc_popart_Builder_addInputTensor_2 =
    R"doc(Add a new input tensor to the model

Parameter ``dataType``:
    The type of the input tensor

Parameter ``shape``:
    The shape of the input tensor

Parameter ``debugContext``:
    Optional debug information

Returns:
    The unique name of the input tensor)doc";

static const char *__doc_popart_Builder_addInputTensorFromParentGraph =
    R"doc(Add a new named input tensor to the model

Parameter ``tensorId``:
    The identifier string of the input tensor. This identifier must
    already exist in the parent GraphProto's name scope and must
    appear topologically before this sub-graph.)doc";

static const char *__doc_popart_Builder_addNodeAttribute =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An int64_t value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_2 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An std::vector<int64_t> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_3 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A float value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_4 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An std::vector<float> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_5 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A std::string value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_6 = R"doc()doc";

static const char *__doc_popart_Builder_addNodeAttribute_7 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An std::vector<std::string> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_8 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An bool value of the attribute to add

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_9 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An constant tensor initializer

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addOutputTensor =
    R"doc(Adds one of the outputs from a node in the graph into the list of
output tensors.)doc";

static const char *__doc_popart_Builder_addUntypedInputTensor =
    R"doc(Add a new input tensor without a type or shape to the model

Parameter ``debugContext``:
    Optional debug information

Returns:
    The unique name of the input tensor)doc";

static const char *__doc_popart_Builder_aiGraphcoreOpset1 =
    R"doc(Return the builder interface for ai.graphcore opset 1)doc";

static const char *__doc_popart_Builder_aiOnnxMlOpset1 =
    R"doc(Return the builder interface for ai.onnx.ml opset 1)doc";

static const char *__doc_popart_Builder_aiOnnxOpset10 =
    R"doc(Return the builder interface for ai.onnx opset 10)doc";

static const char *__doc_popart_Builder_aiOnnxOpset11 =
    R"doc(Return the builder interface for ai.onnx opset 11)doc";

static const char *__doc_popart_Builder_aiOnnxOpset6 =
    R"doc(Return the builder interface for ai.onnx opset 6)doc";

static const char *__doc_popart_Builder_aiOnnxOpset7 =
    R"doc(Return the builder interface for ai.onnx opset 7)doc";

static const char *__doc_popart_Builder_aiOnnxOpset8 =
    R"doc(Return the builder interface for ai.onnx opset 7)doc";

static const char *__doc_popart_Builder_aiOnnxOpset9 =
    R"doc(Return the builder interface for ai.onnx opset 9)doc";

static const char *__doc_popart_Builder_checkpointOutput =
    R"doc(Add checkpoint operations to the model

This is the same as an identity but is recomputeType Checkpoint by
default. Use this to checkpoint a subset of an operation's output
tensors.

Parameter ``nodeOutputNames``:
    Tensors to checkpoint

Returns:
    The checkpointed tensors)doc";

static const char *__doc_popart_Builder_children = R"doc()doc";

static const char *__doc_popart_Builder_clearAttribute =
    R"doc(Unset an attribute that will be set on all subsequent operations)doc";

static const char *__doc_popart_Builder_configure = R"doc()doc";

static const char *__doc_popart_Builder_configure_2 = R"doc()doc";

static const char *__doc_popart_Builder_create =
    R"doc(Create a builder for an ONNX model.)doc";

static const char *__doc_popart_Builder_createFromOnnxModel =
    R"doc(Create a builder which loads a serialized ONNX ModelProto into the
builder and validates it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_createSubgraphBuilder =
    R"doc(Return a Builder for a graph which is nested inside this Builder's
graph.)doc";

static const char *__doc_popart_Builder_customOp = R"doc()doc";

static const char *__doc_popart_Builder_customOp_2 = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns_2 = R"doc()doc";

static const char *__doc_popart_Builder_executionPhase =
    R"doc(Set the execution phase that computes the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node

Parameter ``value``:
    The index of the virtual graph that computes this node)doc";

static const char *__doc_popart_Builder_executionPhase_2 = R"doc()doc";

static const char *__doc_popart_Builder_getAllNodeAttributeNames =
    R"doc(Get all the attribute names from the ONNX node. This functions will
throw an exception if it can't find the unique node.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getAttribute =
    R"doc(Get an attribute that has been set for all subsequent operations)doc";

static const char *__doc_popart_Builder_getAttribute_2 =
    R"doc(Get current attribute value)doc";

static const char *__doc_popart_Builder_getBoolNodeAttribute = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase =
    R"doc(A convenience function for the execution phase attribute)doc";

static const char *__doc_popart_Builder_getExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase_3 = R"doc()doc";

static const char *__doc_popart_Builder_getFloatNodeAttribute =
    R"doc(Get the float value of the attribute for the ONNX node. This functions
will throw an exception if it can't find the unique node or the
attribute does not exist or it has not been set to the float type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getFloatVectorNodeAttribute =
    R"doc(Get the std::vector<float> value of the attribute for the ONNX node.
This functions will throw an exception if it can't find the unique
node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getInputTensorIds =
    R"doc(Return a list of ONNX graph input tensor ids

Returns:
    A vector of input tensor names)doc";

static const char *__doc_popart_Builder_getInt64NodeAttribute =
    R"doc(Get the int64_t value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the int64_t
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getInt64VectorNodeAttribute =
    R"doc(Get the std::vector<int64_t> value of the attribute for the ONNX node.
This functions will throw an exception if it can't find the unique
node or the attribute does not exist or it has not been set to the
std::vector<int64_t> type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getModelProto =
    R"doc(Retrieve the ONNX serialized ModelProto

Returns:
    A serialized ONNX ModelProto)doc";

static const char *__doc_popart_Builder_getNameScope =
    R"doc(Get the current namescope stack using the default delimiter

Parameter ``name``:
    Optional string to concatenate to the end of the stack

Returns:
    A string of the concatenated namescope stack.)doc";

static const char *__doc_popart_Builder_getOutputTensorIds =
    R"doc(Return a list of ONNX graph output tensor ids

Returns:
    A vector of output tensor names)doc";

static const char *__doc_popart_Builder_getPartialsType =
    R"doc(Get the partials type for the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node)doc";

static const char *__doc_popart_Builder_getPipelineStage =
    R"doc(A convenience function for the pipeline stage attribute)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass =
    R"doc(Get whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass_2 =
    R"doc(Get whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getStringNodeAttribute =
    R"doc(Get the std::string value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the std::string
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getStringVectorNodeAttribute =
    R"doc(Get the std::vector<std::string> value of the attribute for the ONNX
node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute)doc";

static const char *__doc_popart_Builder_getTensorDataType =
    R"doc(Return a tensor type from either the input, output, or value_info
lists in the GraphProto

Parameter ``id``:
    Tensor id

Returns:
    A tensor type)doc";

static const char *__doc_popart_Builder_getTensorDtypeString =
    R"doc(Return an ONNX graph tensor type as a lower case string, from either
the input, output, or value_info lists in the GraphProto

Parameter ``id``:
    Tensor id

Returns:
    A lower case string of tensor type)doc";

static const char *__doc_popart_Builder_getTensorShape =
    R"doc(Return an ONNX graph tensor shape, from either the input, output, or
value_info lists in the GraphProto

Parameter ``id``:
    Tensor id

Returns:
    A vector of tensor dimensions)doc";

static const char *__doc_popart_Builder_getTrainableTensorIds =
    R"doc(Return a list of ONNX graph initialized tensor ids

These tensors are stored in the `initialized` section of the ONNX
GraphProto structure.

Returns:
    A vector of tensor names)doc";

static const char *__doc_popart_Builder_getValueTensorIds =
    R"doc(Return a list of ONNX graph value tensor ids

These tensors are stored in the `value_info` section of the ONNX
GraphProto structure.

Returns:
    A vector of output tensor names)doc";

static const char *__doc_popart_Builder_getVirtualGraph =
    R"doc(A convenience function for the virtual graph attribute)doc";

static const char *__doc_popart_Builder_getVirtualGraph_2 =
    R"doc(Get the index of the virtual graph that computes this node. This
applies in a multi IPU system.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getVirtualGraph_3 =
    R"doc(Get the index of the virtual graph that computes this node. This
applies in a multi IPU system.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_hasAttribute = R"doc()doc";

static const char *__doc_popart_Builder_hasAttribute_2 =
    R"doc(Check if attribute is set)doc";

static const char *__doc_popart_Builder_impl = R"doc()doc";

static const char *__doc_popart_Builder_isInitializer =
    R"doc(Returns true if the ONNX tensor is in the initializer list of the
GraphProto

Parameter ``id``:
    Tensor id

Returns:
    A boolean)doc";

static const char *__doc_popart_Builder_loadModelProto =
    R"doc(Load a serialized ONNX ModelProto into the builder and validate it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_nChildren = R"doc()doc";

static const char *__doc_popart_Builder_nodeHasAttribute =
    R"doc(Check whether the ONNX node has an attribute set. This functions will
throw an exception if it can't find the unique node.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_outputTensorLocation = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Builder_popNameScope =
    R"doc(Remove the last entry in the name scope stack)doc";

static const char *__doc_popart_Builder_pushNameScope =
    R"doc(Push a name onto the name scope stack.

The names of tensors and nodes added to the ONNX graph will be
prefixed with a concatenation of the names in the name stack.)doc";

static const char *__doc_popart_Builder_recomputeOutput = R"doc()doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node

Parameter ``value``:
    If the recompute is enabled/disabled)doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass_2 =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node

Parameter ``value``:
    If the recompute is enabled/disabled)doc";

static const char *__doc_popart_Builder_removeNodeAttribute =
    R"doc(Remove an attribute from the ONNX node. This functions will throw an
exception if it can't find the unique node or the attribute does not
exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_reshape_const =
    R"doc(This is a helper function that will add a constant and a reshape using
the provided domain.)doc";

static const char *__doc_popart_Builder_saveInitializersExternally =
    R"doc(The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To prevent this for large models, ONNX tensor data can be
saved separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data. This can be
    an absolute or relative path. If a relative path, when the onnx
    model is saved, external tensor data will be written to a path
    relative to your current working directory.)doc";

static const char *__doc_popart_Builder_saveModelProto =
    R"doc(Save the builder's ONNX ModelProto into the builder and validate it.

Parameter ``fn``:
    The name of a file containing an ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_setAttribute =
    R"doc(Set an attribute that will be set on all subsequent operations)doc";

static const char *__doc_popart_Builder_setAvailableMemoryProportion =
    R"doc(Set the available memory for the given node. Used on the convolution
op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node

Parameter ``availableMemoryProportion``:
    The available memory proportion 0 < x <= 1.)doc";

static const char *__doc_popart_Builder_setGraphName =
    R"doc(Specifies a graph name

Parameter ``name``:
    string to name the graph)doc";

static const char *__doc_popart_Builder_setInplacePreferences = R"doc()doc";

static const char *__doc_popart_Builder_setPartialsType =
    R"doc(Set the partials type for the given node. Used on the convolution op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node

Parameter ``partialsType``:
    The type for the partials. Can be either FLOAT or HALF.)doc";

static const char *__doc_popart_Builder_setSerializeMatMul =
    R"doc(Set the settings for matmuls that should be serialized. This option
will split a matmul into seperate smaller matmuls that will be excuted
in series. This will also serialize the grad operations if training.

Parameter ``nodeOutputNames``:
    Name of the output matmul tensors of the ONNX node

Parameter ``mode``:
    Which dimension of the mat mul to serialize on.

Parameter ``factor``:
    The number of serialised matmuls, must be a factor of the
    dimentions to serialise on.)doc";

static const char *__doc_popart_Builder_virtualGraph =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node

Parameter ``value``:
    The index of the virtual graph that computes this node)doc";

static const char *__doc_popart_Builder_virtualGraph_2 =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node

Parameter ``value``:
    The index of the virtual graph that computes this node)doc";

static const char *__doc_popart_ClipNormSettings =
    R"doc(A data structure used to represent a maximum value constaint on one or
more weights.)doc";

static const char *__doc_popart_ClipNormSettings_ClipNormSettings =
    R"doc(Constructor.

Parameter ``weightIds_``:
    the weight tensor IDs that this constraint applies to.

Parameter ``maxNorm_``:
    the maximum permissible value.)doc";

static const char *__doc_popart_ClipNormSettings_maxNorm = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_eq = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_ne = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_weightIds = R"doc()doc";

static const char *__doc_popart_ConstSGD =
    R"doc(Stochastic Gradient Descent (SGD) optimizer with constant learning
rate, weight decay, loss scaling and clip norm settings (and default
values for momentum, dampening or velocity scaling).

**NOTE**: See SGD for detailed meaning for these parameters.

**NOTE**: This class exists for backwards compatibility with the
Python API and may be removed at some point in the future.)doc";

static const char *__doc_popart_ConstSGD_ConstSGD =
    R"doc(Constructor.

Parameter ``learningRate``:
    a constant learning rate.

Parameter ``weightDecay``:
    a constant weight decay value.

Parameter ``lossScaling``:
    a constant loss scaling value.

Parameter ``clipNormSettings``:
    a vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_DataFlow =
    R"doc(This class specifies parameters for host-device data streams. The
parameters are used to control the amount input data processed each
step (that is: each Session::run call) determines how data is returned
to the user.

See also: AnchorReturnType, #AnchorReturnTypeId.)doc";

static const char *__doc_popart_DataFlow_DataFlow =
    R"doc(Default constructor, sets `batchesPerStep` to 0 and does not have any
anchors.)doc";

static const char *__doc_popart_DataFlow_DataFlow_2 =
    R"doc(Construct DataFlow instance without anchor tensors.

Parameter ``batchesPerStep``:
    - the number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.)doc";

static const char *__doc_popart_DataFlow_DataFlow_3 =
    R"doc(Constructor DataFlow instance with anchor tensors.

Parameter ``batchesPerStep``:
    the number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorMap``:
    a mapping from output tensor TensorId to AnchorReturnType
    indicating the strategy with which to write the anchor tensor
    values to the IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_4 =
    R"doc(Constructor DataFlow instance with anchor tensors.

Parameter ``batchesPerStep``:
    the number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorTensorIds``:
    the tensor ID of anchor tensors.

Parameter ``anchorReturnType``:
    the strategy with which to write anchor tensor values to the
    IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_5 = R"doc()doc";

static const char *__doc_popart_DataFlow_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_art = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_getAnchorMap = R"doc()doc";

static const char *__doc_popart_DataFlow_hash = R"doc()doc";

static const char *__doc_popart_DataFlow_isAnchored = R"doc()doc";

static const char *__doc_popart_DataFlow_isBatchCountingRequired = R"doc()doc";

static const char *__doc_popart_DataFlow_isValidAnchorReturnPeriod =
    R"doc()doc";

static const char *__doc_popart_DataFlow_m_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_nAnchors = R"doc()doc";

static const char *__doc_popart_DataFlow_numOutFetchesPerRepl = R"doc()doc";

static const char *__doc_popart_DataFlow_operator_assign = R"doc()doc";

static const char *__doc_popart_DataFlow_rps = R"doc()doc";

static const char *__doc_popart_DataFlow_s_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_v_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_v_rps = R"doc()doc";

static const char *__doc_popart_DataType = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Always = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Never = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_OnDemand = R"doc()doc";

static const char *__doc_popart_DeviceInfo = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_2 = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_DeviceInfo = R"doc()doc";

static const char *__doc_popart_DeviceInfo_attach =
    R"doc(Attach to the IPU.

Returns:
    Returns true if successfully attaches to the device)doc";

static const char *__doc_popart_DeviceInfo_attachTimeout = R"doc()doc";

static const char *__doc_popart_DeviceInfo_canCompileOffline = R"doc()doc";

static const char *__doc_popart_DeviceInfo_connectionType = R"doc()doc";

static const char *__doc_popart_DeviceInfo_detach =
    R"doc(Detach from the IPU.)doc";

static const char *__doc_popart_DeviceInfo_flags = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getConnectionType =
    R"doc(Get the connection type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getDriverIds = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getId =
    R"doc(Get the device id.)doc";

static const char *__doc_popart_DeviceInfo_getNumIpus =
    R"doc(Get the number of IPUs in the device.)doc";

static const char *__doc_popart_DeviceInfo_getNumWorkerContexts =
    R"doc(Get the number of worker contexts per tile.)doc";

static const char *__doc_popart_DeviceInfo_getOnDemandAttachTimeout =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_getOptionFlags = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getTarget = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getTilesPerIPU =
    R"doc(Get the number of tiles per IPU.)doc";

static const char *__doc_popart_DeviceInfo_getType =
    R"doc(Get the type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getVersion =
    R"doc(Get the version of the software on the IPU.)doc";

static const char *__doc_popart_DeviceInfo_provider = R"doc()doc";

static const char *__doc_popart_DeviceInfo_setOnDemandAttachTimeout =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_toString =
    R"doc(Return a description of the device.)doc";

static const char *__doc_popart_DeviceInfo_tryAttachUntilTimeout = R"doc()doc";

static const char *__doc_popart_DeviceInfo_type = R"doc()doc";

static const char *__doc_popart_DeviceManager =
    R"doc(A class to manage devices.)doc";

static const char *__doc_popart_DeviceManager_acquireAvailableDevice =
    R"doc(Finds the first available hardware device, with a certain number of
IPUs. This method will attach to the device.

Parameter ``numIpus``:
    The number of IPUs on the device [=1]

Parameter ``tilesPerIPU``:
    The number of tiles per IPU (0 will match any number) [=0]

Returns:
    A device, which can be used with a session. Will return nullptr if
    no device is available)doc";

static const char *__doc_popart_DeviceManager_acquireDeviceById =
    R"doc(Allocates the hardware device by id. This id can be found running 'gc-
info -l'. This method will attach to the device.

Parameter ``id``:
    The index of the IPU to be used

Returns:
    A device. Will return nullptr if the device is not available)doc";

static const char *__doc_popart_DeviceManager_attachTimeout = R"doc()doc";

static const char *__doc_popart_DeviceManager_createCpuDevice =
    R"doc(Create a 'simulated' CPU device.

Returns:
    A device)doc";

static const char *__doc_popart_DeviceManager_createDeviceManager =
    R"doc(Accessor for the device manager.

Returns:
    A reference to the DeviceManager)doc";

static const char *__doc_popart_DeviceManager_createIpuModelDevice =
    R"doc(Create a 'simulated' IPU Model device. The following options are
supported:

* ``numIPUs``: The number of IPUs to simulate [=1] * ``ge``: The
number of tiles per IPU [=defaultFewTiles] * ``compileIPUCode``:
Whether or not to compile real IPU code for modelling

Parameter ``options``:
    Configuration settings for the IPU Model

Returns:
    A device)doc";

static const char *__doc_popart_DeviceManager_createOfflineIPUDevice =
    R"doc(Create a device resembling an IPU for offline compilation The
following options are supported:

* ``numIPUs``: The number of IPUs to compile for * ``ge``: The number
of tiles per IPU [=defaultManyTiles] * ``ipuVersion``: The ipu
architecture [="ipu1"] * ``syncPattern``: The sync pattern to use:
full/singlePipline/replicaAndLadder, defaults to full

Parameter ``options``:
    Configuration settings for the IPU Model

Returns:
    A device)doc";

static const char *__doc_popart_DeviceManager_createSimDevice = R"doc()doc";

static const char *__doc_popart_DeviceManager_enumerateDevices =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``pattern``:
    Sync pattern.

Parameter ``numIpus``:
    Number of IPUs to request.

Parameter ``deviceType``:
    Type of device required.

Parameter ``tilesPerIPU``:
    The number of tiles per ipu required.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_getDevice =
    R"doc(Get the Device object of a device by ID.

Parameter ``syncPattern``:
    Sync pattern

Parameter ``deviceManagerId``:
    Number of IPUs to request.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_providers = R"doc()doc";

static const char *__doc_popart_DeviceManager_registerDeviceProvider =
    R"doc(Used to register a device provider.

Parameter ``provider``:
    A provider)doc";

static const char *__doc_popart_DeviceManager_setOnDemandAttachTimeout =
    R"doc(If unable to attach to a device on first try, the attach timeout set
here is the length of time (in seconds) that the DeviceManager will
wait to try and attach. Note: this only takes effect when trying to
attach with a DeviceConnectionType::OnDemand DeviceConnectionType.

Parameter ``seconds``:
    The attach timeout in seconds)doc";

static const char *__doc_popart_DeviceProvider =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_2 =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_createHostDevice =
    R"doc(Create a host device for testing)doc";

static const char *__doc_popart_DeviceProvider_enumerate =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``devices``:
    Devices to get

Parameter ``requiredNumIPUs``:
    Number of IPUs to request.

Parameter ``syncPattern``:
    Sync pattern

Parameter ``requiredTilesPerIPU``:
    Number of tiles per IPU to request.)doc";

static const char *__doc_popart_DeviceProvider_getDevice = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_First = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_Random = R"doc()doc";

static const char *__doc_popart_DeviceType = R"doc()doc";

static const char *__doc_popart_DeviceType_Cpu = R"doc()doc";

static const char *__doc_popart_DeviceType_Ipu = R"doc()doc";

static const char *__doc_popart_DeviceType_IpuModel = R"doc()doc";

static const char *__doc_popart_DeviceType_OfflineIpu = R"doc()doc";

static const char *__doc_popart_DeviceType_Sim = R"doc()doc";

static const char *__doc_popart_DomainOpSet = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet_2 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_DomainOpSet_impl = R"doc()doc";

static const char *__doc_popart_DotCheck =
    R"doc(Enum type used to identify at which stages of IR construction to
export .dot files.)doc";

static const char *__doc_popart_DotCheck_Bwd0 =
    R"doc(Generate graph after backwards construction.)doc";

static const char *__doc_popart_DotCheck_Final =
    R"doc(Generate graph after running aliasing patterns (the final IR).)doc";

static const char *__doc_popart_DotCheck_Fwd0 =
    R"doc(Generate graph after construction of the forward pass.)doc";

static const char *__doc_popart_DotCheck_Fwd1 =
    R"doc(Generate graph after running pre-aliasing patterns.)doc";

static const char *__doc_popart_DotCheck_N =
    R"doc(The number of DotCheck values.)doc";

static const char *__doc_popart_DotCheck_PreAlias =
    R"doc(Generate graph after all transformations, patterns, except the
aliasing.)doc";

static const char *__doc_popart_ErrorSource = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart_internal = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplar = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplibs = R"doc()doc";

static const char *__doc_popart_ErrorSource_unknown = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule =
    R"doc(Enum type to specify when to load tensors.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_N =
    R"doc(The number of ExecutionPhaseIOSchedule values.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_OnDemand =
    R"doc(Load tensors just before they are required)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_Preload =
    R"doc(Preload tensors in previous phase for use in current phase)doc";

static const char *__doc_popart_ExecutionPhaseSchedule =
    R"doc(Enum type to specify the order of processing optimizer operations for
different weights of the same execution phase.

The steps for phased execution consists of: - Copy to IO tiles if
necessary (1) - Run collective operations if necessary (2) - Load
optimizer state (3) - Update optimizer state (4) - Apply optimizer (5)
- Store updated tensor if necessary (6))doc";

static const char *__doc_popart_ExecutionPhaseSchedule_Batch =
    R"doc(Process above steps for all weights together, in a way that maximises
overlap potential between compute and exchange (for example: 333, 111,
222, 444, 555, 666).)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_BatchClusteredIO =
    R"doc(Process above steps for all weights together, in a way that maximises
overlap potential between compute and exchange, and maximise stream
copy merges by keeping RemoteLoad/RemoteStore operations clustered
(for example: 333, 111, 222, 444, 555, 666).)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_Interleaving =
    R"doc(Process above steps for one weight at a time (for example: 123456,
123456, 123456). The scheduler may interleave these steps.)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_N =
    R"doc(The number of ExecutionPhaseSchedule values.)doc";

static const char *__doc_popart_ExecutionPhaseSettings =
    R"doc(A structure containing ExecutionPhase settings.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings =
    R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings_2 = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_accumulatorIOSchedule =
    R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_activationIOSchedule =
    R"doc(The execution phase IO schedule for activation and gradient tensors.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_operator_assign =
    R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_optimizerStateIOSchedule = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_phases =
    R"doc(Number of ExecutionPhases for the whole model)doc";

static const char *__doc_popart_ExecutionPhaseSettings_schedule = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_stages =
    R"doc(Number of overlapping stages 1: Parallel streaming memory, default for
1 IPU / replica 2: PingPong between 2 IPUs, default for >= 2 IPUs /
replica)doc";

static const char *__doc_popart_ExecutionPhaseSettings_weightIOSchedule =
    R"doc(The execution phase IO schedule for weight tensors.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy =
    R"doc(Enum type that describes how copies for inputs and outputs for
subgraphs are lowered. Currently this only affects subgraphs
associated with CallOps.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_JustInTime =
    R"doc(Copy inputs just before they are consumed and copy outputs as soon as
they are produced. With this strategy subgraphs may be lowered into
multiple Poplar functions.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_OnEnterAndExit =
    R"doc(Copy all inputs before the start of the subgraph, copy all outputs
after all ops in the subgraph. With this strategy subgraphs will
always map to a single Poplar function.)doc";

static const char *__doc_popart_GradNonGradPair = R"doc()doc";

static const char *__doc_popart_GradNonGradPair_GradNonGradPair = R"doc()doc";

static const char *__doc_popart_GradNonGradPair_GradNonGradPair_2 = R"doc()doc";

static const char *__doc_popart_GradNonGradPair_grad = R"doc()doc";

static const char *__doc_popart_GradNonGradPair_nongrad = R"doc()doc";

static const char *__doc_popart_GraphTransformer = R"doc()doc";

static const char *__doc_popart_GraphTransformerImpl = R"doc()doc";

static const char *__doc_popart_GraphTransformer_GraphTransformer = R"doc()doc";

static const char
    *__doc_popart_GraphTransformer_convertAllFixedPointInitializersToConstants =
        R"doc(Convert all of the fixed-point initializers into ONNX Constant Nodes)doc";

static const char *__doc_popart_GraphTransformer_convertBFloats16ToFloat32 =
    R"doc(Convert the graph from BFloat16 to Float32)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToFloats =
    R"doc(Convert the graph from float64 to float32)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToHalfs =
    R"doc(Convert the graph from float64 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertFloatsToHalfs =
    R"doc(Convert the graph from float32 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertINT16ToINT32 =
    R"doc(Convert the graph from int16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertINT64ToINT32 =
    R"doc(Convert the graph from int64 to int32

Parameter ``clip``:
    If tensor data are outside of the numerical range expressible by
    int32, clip to max and min numeric limits)doc";

static const char *__doc_popart_GraphTransformer_convertINT8ToINT32 =
    R"doc(Convert the graph from int8 to int32)doc";

static const char
    *__doc_popart_GraphTransformer_convertInitializersToConstants =
        R"doc(Convert the given list of initializers into ONNX Constant Nodes

Parameter ``ids``:
    A list of initializer names)doc";

static const char *__doc_popart_GraphTransformer_convertUINT16ToINT32 =
    R"doc(Convert the graph from uint16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertUINT8ToINT32 =
    R"doc(Convert the graph from uint8 to int32)doc";

static const char *__doc_popart_GraphTransformer_getModelProto = R"doc()doc";

static const char *__doc_popart_GraphTransformer_impl = R"doc()doc";

static const char *__doc_popart_GraphTransformer_prepareNodesForTraining =
    R"doc(Some ONNX Operators are different between train and test modes An
example is BatchNormalization, which has 1 output in test mode and 5
outputs in train mode This function changes the Nodes to be of the
training variety)doc";

static const char *__doc_popart_GraphTransformer_removeUnusedInputs =
    R"doc(Inputs which are not connected to any Node are removed)doc";

static const char *__doc_popart_GraphTransformer_saveInitializersExternally =
    R"doc(The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To prevent this for large models, ONNX tensor data can be
saved separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data.)doc";

static const char *__doc_popart_IStepIO = R"doc()doc";

static const char *__doc_popart_IdentityGradOp = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_aliases = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_IdentityLossGradOp =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_canBeReplacedByIdentity =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getReductionType =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getSubgraphValue =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradOutToNonGradIn =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_outShape = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_reduction_type = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_setup = R"doc()doc";

static const char *__doc_popart_IdentityLossOp = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_IdentityLossOp = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_canBeReplacedByIdentity =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getGradOps = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getReductionType = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_reduction_type = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_setup = R"doc()doc";

static const char *__doc_popart_IdentityOp = R"doc()doc";

static const char *__doc_popart_IdentityOp_IdentityOp = R"doc()doc";

static const char *__doc_popart_IdentityOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityOp_getGradOps = R"doc()doc";

static const char *__doc_popart_IdentityOp_getInplaceVariant = R"doc()doc";

static const char *__doc_popart_IdentityOp_inplacePriorityDefault = R"doc()doc";

static const char *__doc_popart_InferenceSession = R"doc()doc";

static const char *__doc_popart_InferenceSession_InferenceSession = R"doc()doc";

static const char *__doc_popart_InferenceSession_configureFromOnnx =
    R"doc()doc";

static const char *__doc_popart_InferenceSession_createFromOnnxModel =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for inference

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches

Parameter ``userOptions``:
    String to configure session options

Parameter ``patterns``:
    Optimization patterns to apply)doc";

static const char *__doc_popart_InitOp = R"doc()doc";

static const char *__doc_popart_InitOp_InitOp = R"doc()doc";

static const char *__doc_popart_InitOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_InitOp_batch_axis = R"doc()doc";

static const char *__doc_popart_InitOp_canShard = R"doc()doc";

static const char *__doc_popart_InitOp_clone = R"doc()doc";

static const char *__doc_popart_InitOp_getInitType = R"doc()doc";

static const char *__doc_popart_InitOp_getOutBatchAxis = R"doc()doc";

static const char *__doc_popart_InitOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_InitOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorInfo = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorType = R"doc()doc";

static const char *__doc_popart_InitOp_init_type = R"doc()doc";

static const char *__doc_popart_InitOp_isOutlineable = R"doc()doc";

static const char *__doc_popart_InitOp_setup = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_info = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_type = R"doc()doc";

static const char *__doc_popart_InitType = R"doc()doc";

static const char *__doc_popart_InitType_NoInit = R"doc()doc";

static const char *__doc_popart_InitType_Zero = R"doc()doc";

static const char *__doc_popart_Instrumentation =
    R"doc(Enum type used to specify an instrumentation type.)doc";

static const char *__doc_popart_Instrumentation_Inner =
    R"doc(Inner loop instrumentation, graph per IPU.)doc";

static const char *__doc_popart_Instrumentation_N =
    R"doc(The number of Instrumentations values.)doc";

static const char *__doc_popart_Instrumentation_Outer =
    R"doc(Outer loop instrumentation, graph over all IPUs.)doc";

static const char *__doc_popart_Ir = R"doc()doc";

static const char *__doc_popart_IrBundle = R"doc()doc";

static const char *__doc_popart_IrBundle_IrBundle = R"doc()doc";

static const char *__doc_popart_IrBundle_dataFlow = R"doc()doc";

static const char *__doc_popart_IrBundle_deviceInfo = R"doc()doc";

static const char *__doc_popart_IrBundle_inputShapeInfo = R"doc()doc";

static const char *__doc_popart_IrBundle_loss = R"doc()doc";

static const char *__doc_popart_IrBundle_modelProto = R"doc()doc";

static const char *__doc_popart_IrBundle_optimizer = R"doc()doc";

static const char *__doc_popart_IrBundle_patterns = R"doc()doc";

static const char *__doc_popart_IrBundle_userOptions = R"doc()doc";

static const char *__doc_popart_IrSerializationFormat =
    R"doc(Enum type used to specify a serialization format.)doc";

static const char *__doc_popart_IrSerializationFormat_JSON =
    R"doc(JavaScript Object Notation (JSON).)doc";

static const char *__doc_popart_Ir_ExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Inference = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Training = R"doc()doc";

static const char *__doc_popart_Ir_Ir = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_SavedInfo = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_SavedInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_SavedInfo_3 = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_deserialize = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_irHash = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_operator_eq = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_serialize = R"doc()doc";

static const char *__doc_popart_Ir_SavedInfo_toString = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat_JSON = R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor = R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_addOp = R"doc()doc";

static const char *__doc_popart_Ir_additionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_append = R"doc()doc";

static const char *__doc_popart_Ir_applyInplacePattern = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPattern = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPatterns = R"doc()doc";

static const char *__doc_popart_Ir_applyTransform = R"doc()doc";

static const char *__doc_popart_Ir_applyUpdateInplacePrioritiesForIpu =
    R"doc()doc";

static const char *__doc_popart_Ir_autoRecomputationEnabled = R"doc()doc";

static const char *__doc_popart_Ir_canInfer = R"doc()doc";

static const char *__doc_popart_Ir_canTrain = R"doc()doc";

static const char *__doc_popart_Ir_compareWithSavedHash = R"doc()doc";

static const char *__doc_popart_Ir_confirmConstIds = R"doc()doc";

static const char *__doc_popart_Ir_confirmNoReservedIds = R"doc()doc";

static const char *__doc_popart_Ir_confirmNonReservedId = R"doc()doc";

static const char *__doc_popart_Ir_constructBackwards = R"doc()doc";

static const char *__doc_popart_Ir_constructForwards = R"doc()doc";

static const char *__doc_popart_Ir_constructFromOnnxGraph = R"doc()doc";

static const char *__doc_popart_Ir_constructedBackwards = R"doc()doc";

static const char *__doc_popart_Ir_constructedFinalLoss = R"doc()doc";

static const char *__doc_popart_Ir_containsInitialisers = R"doc()doc";

static const char *__doc_popart_Ir_containsTensor = R"doc()doc";

static const char *__doc_popart_Ir_createConcatTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createGraph = R"doc()doc";

static const char *__doc_popart_Ir_createIntermediateTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createSliceTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createUniqueSubgraphId = R"doc()doc";

static const char *__doc_popart_Ir_dataFlow = R"doc()doc";

static const char *__doc_popart_Ir_dataStreamTensors = R"doc()doc";

static const char *__doc_popart_Ir_decomposedOptimizers = R"doc()doc";

static const char *__doc_popart_Ir_deviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_dotCheckpoint = R"doc()doc";

static const char *__doc_popart_Ir_enableTransform = R"doc()doc";

static const char *__doc_popart_Ir_ensureOptimizerTensorCreated = R"doc()doc";

static const char *__doc_popart_Ir_executionMode = R"doc()doc";

static const char *__doc_popart_Ir_executionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_finalLossId = R"doc()doc";

static const char *__doc_popart_Ir_finalLossOpId = R"doc()doc";

static const char *__doc_popart_Ir_foldConstants = R"doc()doc";

static const char *__doc_popart_Ir_getAccumulateOuterFragmentBinConstraints =
    R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAllGraphs = R"doc()doc";

static const char *__doc_popart_Ir_getAllOps = R"doc()doc";

static const char *__doc_popart_Ir_getAllRemoteBufferInfos = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrOpsCounter = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementRandomReferenceId =
    R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementSeedModifier = R"doc()doc";

static const char *__doc_popart_Ir_getDataFlow = R"doc()doc";

static const char *__doc_popart_Ir_getDefaultOpsetVersion = R"doc()doc";

static const char *__doc_popart_Ir_getDeviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossId = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossOpId = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossPipelineStage = R"doc()doc";

static const char *__doc_popart_Ir_getGradSumOpNamePrefix = R"doc()doc";

static const char *__doc_popart_Ir_getGraph = R"doc()doc";

static const char *__doc_popart_Ir_getGraphInputIds = R"doc()doc";

static const char *__doc_popart_Ir_getGraphSchedule = R"doc()doc";

static const char *__doc_popart_Ir_getGraphs = R"doc()doc";

static const char *__doc_popart_Ir_getHash = R"doc()doc";

static const char *__doc_popart_Ir_getInputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_getIrBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMaxVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Ir_getModel = R"doc()doc";

static const char *__doc_popart_Ir_getModelInputIds = R"doc()doc";

static const char *__doc_popart_Ir_getNumPipelineStages = R"doc()doc";

static const char *__doc_popart_Ir_getOpSchedule = R"doc()doc";

static const char *__doc_popart_Ir_getOpSetVersionFromModel = R"doc()doc";

static const char *__doc_popart_Ir_getOpsCounter = R"doc()doc";

static const char *__doc_popart_Ir_getOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_getOrSetRandomReferenceTensor = R"doc()doc";

static const char *__doc_popart_Ir_getPatternLevelStr = R"doc()doc";

static const char *__doc_popart_Ir_getPatterns = R"doc()doc";

static const char *__doc_popart_Ir_getPopartCachePath = R"doc()doc";

static const char *__doc_popart_Ir_getRemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_Ir_getRequiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_getRootInputsToOp = R"doc()doc";

static const char *__doc_popart_Ir_getSessionOptions = R"doc()doc";

static const char *__doc_popart_Ir_getSubgraphAnchorPlaceholder = R"doc()doc";

static const char *__doc_popart_Ir_getTensor = R"doc()doc";

static const char *__doc_popart_Ir_getTensorIds = R"doc()doc";

static const char *__doc_popart_Ir_getTensors = R"doc()doc";

static const char *__doc_popart_Ir_getTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_getTrainTargetOps = R"doc()doc";

static const char *__doc_popart_Ir_getVirtualGraphIdFromTensorProducers =
    R"doc()doc";

static const char *__doc_popart_Ir_graphs = R"doc()doc";

static const char *__doc_popart_Ir_growCopyVarUpdateOp = R"doc()doc";

static const char *__doc_popart_Ir_growGradOps = R"doc()doc";

static const char *__doc_popart_Ir_growGradSumOp = R"doc()doc";

static const char *__doc_popart_Ir_growGradientVarUpdateOp = R"doc()doc";

static const char *__doc_popart_Ir_growLossGradients = R"doc()doc";

static const char *__doc_popart_Ir_growVarUpdateOpInternal = R"doc()doc";

static const char *__doc_popart_Ir_hasConstructedBackwards = R"doc()doc";

static const char *__doc_popart_Ir_hasDecomposedOptimizers = R"doc()doc";

static const char *__doc_popart_Ir_hasGraph = R"doc()doc";

static const char *__doc_popart_Ir_hasRandomOps = R"doc()doc";

static const char *__doc_popart_Ir_hash = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched_2 = R"doc()doc";

static const char *__doc_popart_Ir_initRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_inputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_intermediate_tensor_counter = R"doc()doc";

static const char *__doc_popart_Ir_irBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_isAnchored = R"doc()doc";

static const char *__doc_popart_Ir_isCandidateForConstExprFolding = R"doc()doc";

static const char *__doc_popart_Ir_isConsumedByOpOfType = R"doc()doc";

static const char *__doc_popart_Ir_isPatternsLevel = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared_2 = R"doc()doc";

static const char *__doc_popart_Ir_isSchedulable = R"doc()doc";

static const char *__doc_popart_Ir_isTesting = R"doc()doc";

static const char *__doc_popart_Ir_isTraining = R"doc()doc";

static const char *__doc_popart_Ir_logIr = R"doc()doc";

static const char *__doc_popart_Ir_mergeRandomReferenceIds = R"doc()doc";

static const char *__doc_popart_Ir_onnxModel = R"doc()doc";

static const char *__doc_popart_Ir_opAndRootInputs = R"doc()doc";

static const char *__doc_popart_Ir_opsCounter = R"doc()doc";

static const char *__doc_popart_Ir_opsOfType = R"doc()doc";

static const char *__doc_popart_Ir_optimizer = R"doc()doc";

static const char *__doc_popart_Ir_optimizerTensors = R"doc()doc";

static const char *__doc_popart_Ir_patterns = R"doc()doc";

static const char *__doc_popart_Ir_prepare = R"doc()doc";

static const char *__doc_popart_Ir_prepareComplete = R"doc()doc";

static const char *__doc_popart_Ir_prepareImpl = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceId = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceTensorMap = R"doc()doc";

static const char *__doc_popart_Ir_registerInputTensors = R"doc()doc";

static const char *__doc_popart_Ir_remoteBufferInfoMap = R"doc()doc";

static const char *__doc_popart_Ir_removeGraph = R"doc()doc";

static const char *__doc_popart_Ir_removeIsolatedTensors = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Ir_saveHash = R"doc()doc";

static const char *__doc_popart_Ir_seedModifier = R"doc()doc";

static const char *__doc_popart_Ir_serialise = R"doc()doc";

static const char *__doc_popart_Ir_setDataFlow = R"doc()doc";

static const char *__doc_popart_Ir_setDeviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_setExternalTensorDataInfo = R"doc()doc";

static const char *__doc_popart_Ir_setFinalLoss = R"doc()doc";

static const char *__doc_popart_Ir_setHash = R"doc()doc";

static const char *__doc_popart_Ir_setInputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_setIrBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_setIsPrepared = R"doc()doc";

static const char *__doc_popart_Ir_setNEdgesToLoss = R"doc()doc";

static const char *__doc_popart_Ir_setOnnxModel = R"doc()doc";

static const char *__doc_popart_Ir_setOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_setPatterns = R"doc()doc";

static const char *__doc_popart_Ir_setRemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_Ir_setRequiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_setUserOptions = R"doc()doc";

static const char *__doc_popart_Ir_step = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_subgraph_id_counter = R"doc()doc";

static const char *__doc_popart_Ir_syntheticDataMode = R"doc()doc";

static const char *__doc_popart_Ir_tensorExistsInInitialisers = R"doc()doc";

static const char *__doc_popart_Ir_transformEnableMap = R"doc()doc";

static const char *__doc_popart_Ir_unsetAllVirtualGraphIds = R"doc()doc";

static const char *__doc_popart_Ir_updateAliases = R"doc()doc";

static const char *__doc_popart_Ir_updateOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_updateVertices = R"doc()doc";

static const char *__doc_popart_Ir_useSyntheticData = R"doc()doc";

static const char *__doc_popart_Ir_userOptions = R"doc()doc";

static const char *__doc_popart_Ir_usingEngineCache = R"doc()doc";

static const char *__doc_popart_Ir_validateAnchors = R"doc()doc";

static const char *__doc_popart_Ir_verifyConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyConstExprFolding = R"doc()doc";

static const char *__doc_popart_Ir_verifyDistributedReplicatedGraphSettings =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyExecutionPhaseSettings = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpInputConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpOutputConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyPipelineSettings = R"doc()doc";

static const char *__doc_popart_Ir_verifyRecomputeAttributes = R"doc()doc";

static const char *__doc_popart_Ir_verifySubgraphs = R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorConsumerConnectivity =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorIds = R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorProducerConnectivity =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVertexAttributesOnlyInMain =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVirtualGraphIds = R"doc()doc";

static const char *__doc_popart_Ir_verifyVirualGraphIdsNotInitialized =
    R"doc()doc";

static const char *__doc_popart_Ir_virtualGraphsEnabled = R"doc()doc";

static const char *__doc_popart_L1GradOp = R"doc()doc";

static const char *__doc_popart_L1GradOp_L1GradOp = R"doc()doc";

static const char *__doc_popart_L1GradOp_canShard = R"doc()doc";

static const char *__doc_popart_L1GradOp_clone = R"doc()doc";

static const char *__doc_popart_L1GradOp_getFwdActInIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getGradInIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getLambda = R"doc()doc";

static const char *__doc_popart_L1GradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getReductionType = R"doc()doc";

static const char *__doc_popart_L1GradOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_L1GradOp_lambda = R"doc()doc";

static const char *__doc_popart_L1GradOp_reduction = R"doc()doc";

static const char *__doc_popart_L1GradOp_setup = R"doc()doc";

static const char *__doc_popart_L1Op = R"doc()doc";

static const char *__doc_popart_L1Op_L1Op = R"doc()doc";

static const char *__doc_popart_L1Op_canShard = R"doc()doc";

static const char *__doc_popart_L1Op_clone = R"doc()doc";

static const char *__doc_popart_L1Op_getGradOps = R"doc()doc";

static const char *__doc_popart_L1Op_getInIndex = R"doc()doc";

static const char *__doc_popart_L1Op_getLambda = R"doc()doc";

static const char *__doc_popart_L1Op_getOutIndex = R"doc()doc";

static const char *__doc_popart_L1Op_getReductionType = R"doc()doc";

static const char *__doc_popart_L1Op_getShardReductionType = R"doc()doc";

static const char *__doc_popart_L1Op_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_L1Op_lambda = R"doc()doc";

static const char *__doc_popart_L1Op_reduction = R"doc()doc";

static const char *__doc_popart_L1Op_setup = R"doc()doc";

static const char *__doc_popart_MergeVarUpdateType =
    R"doc(Enum type used to specify which `VarUpdate` ops to merge.)doc";

static const char *__doc_popart_MergeVarUpdateType_All =
    R"doc(Merge all VarUpdate Ops into as few groups as possible. This is a good
choice when memory is not a constraint.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoLoose =
    R"doc(Merge into groups while attempting not to increase maximum variable
liveness, and also not slice tensor variables so they they will need
to be processed by different `VarUpdate` ops.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoTight =
    R"doc(Merge into groups, so that VarUpdateOps process tensors of exactly
`mergeVarUpdateMemThreshold` in size.)doc";

static const char *__doc_popart_MergeVarUpdateType_N =
    R"doc(The number of MergeVarUpdateTypes values.)doc";

static const char *__doc_popart_MergeVarUpdateType_None =
    R"doc(Do not merge VarUpdate Ops.)doc";

static const char *__doc_popart_NllGradOp = R"doc()doc";

static const char *__doc_popart_NllGradOp_NllGradOp = R"doc()doc";

static const char *__doc_popart_NllGradOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_NllGradOp_clone = R"doc()doc";

static const char *__doc_popart_NllGradOp_getGradInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLabelInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLossTensorId = R"doc()doc";

static const char *__doc_popart_NllGradOp_getOptionalIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getProbsInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getReductionType = R"doc()doc";

static const char *__doc_popart_NllGradOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_NllGradOp_hasIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_ignoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_lossId = R"doc()doc";

static const char *__doc_popart_NllGradOp_reduction = R"doc()doc";

static const char *__doc_popart_NllGradOp_setup = R"doc()doc";

static const char *__doc_popart_NllOp = R"doc()doc";

static const char *__doc_popart_NllOp_NllOp = R"doc()doc";

static const char *__doc_popart_NllOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_NllOp_canShard = R"doc()doc";

static const char *__doc_popart_NllOp_clone = R"doc()doc";

static const char *__doc_popart_NllOp_getGradOps = R"doc()doc";

static const char *__doc_popart_NllOp_getIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getLabelInIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getOptionalIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getProbsInIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getReductionType = R"doc()doc";

static const char *__doc_popart_NllOp_getShardReductionType = R"doc()doc";

static const char *__doc_popart_NllOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_NllOp_hasIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_ignoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability_2 = R"doc()doc";

static const char *__doc_popart_NllOp_reduction = R"doc()doc";

static const char *__doc_popart_NllOp_setup = R"doc()doc";

static const char *__doc_popart_OpCreator = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_OpCreatorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_attributes = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputData = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue_2 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensor = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorData = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getOutputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_inputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_opid = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_outputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_settings = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_2 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_3 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_4 = R"doc()doc";

static const char *__doc_popart_OpDefinition = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_Attribute = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_supportedValuesRegex =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Input = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_Input = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_constant = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_name = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_supportedTensors =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition = R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_Output = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_name = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_supportedTensors =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_attributes = R"doc()doc";

static const char *__doc_popart_OpDefinition_inputs = R"doc()doc";

static const char *__doc_popart_OpDefinition_outputs = R"doc()doc";

static const char *__doc_popart_OpGradRegistry = R"doc()doc";

static const char *__doc_popart_OpGradRegistry_complete = R"doc()doc";

static const char *__doc_popart_OpGradRegistry_insert = R"doc()doc";

static const char *__doc_popart_OpGradRegistry_partial = R"doc()doc";

static const char *__doc_popart_OpGradRegistry_popComplete = R"doc()doc";

static const char *__doc_popart_OpManager = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_complexFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_details = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getComplexFactory =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getSimpleFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_hasComplexFactory =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_id = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_isPublic = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_simpleFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpManager = R"doc()doc";

static const char *__doc_popart_OpManager_checkOpVersionAgainstOpset =
    R"doc()doc";

static const char *__doc_popart_OpManager_create = R"doc()doc";

static const char *__doc_popart_OpManager_create_2 = R"doc()doc";

static const char *__doc_popart_OpManager_createOp = R"doc()doc";

static const char *__doc_popart_OpManager_createOp_2 = R"doc()doc";

static const char *__doc_popart_OpManager_createOpInGraph = R"doc()doc";

static const char *__doc_popart_OpManager_createOpWithInputs = R"doc()doc";

static const char *__doc_popart_OpManager_findOpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_getAttributesFromAnyMap = R"doc()doc";

static const char *__doc_popart_OpManager_getInstance = R"doc()doc";

static const char *__doc_popart_OpManager_getOpVersionFromOpSet = R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperations = R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperationsDefinition =
    R"doc()doc";

static const char *__doc_popart_OpManager_getUnsupportedOperations =
    R"doc()doc";

static const char *__doc_popart_OpManager_opMap = R"doc()doc";

static const char *__doc_popart_OpManager_registerOp = R"doc()doc";

static const char *__doc_popart_Optimizer = R"doc()doc";

static const char *__doc_popart_OptimizerReductionType = R"doc()doc";

static const char *__doc_popart_OptimizerReductionType_AcclReduce = R"doc()doc";

static const char *__doc_popart_OptimizerReductionType_AccumReduce =
    R"doc()doc";

static const char *__doc_popart_OptimizerReductionType_GradReduce = R"doc()doc";

static const char *__doc_popart_OptimizerReductionType_None = R"doc()doc";

static const char *__doc_popart_OptimizerType = R"doc()doc";

static const char *__doc_popart_OptimizerType_Adam = R"doc()doc";

static const char *__doc_popart_OptimizerType_Adaptive = R"doc()doc";

static const char *__doc_popart_OptimizerType_NTYPES = R"doc()doc";

static const char *__doc_popart_OptimizerType_SGD = R"doc()doc";

static const char *__doc_popart_OptimizerValue =
    R"doc(A class used to represent values of hyper parameters.)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue =
    R"doc(Equivalent to OptimizerValue(0, false).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_2 =
    R"doc(Equivalent to OptimizerValue(v, true).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_3 =
    R"doc(Constructor.

Parameter ``v``:
    the current value of the hyper parameter.

Parameter ``c``:
    a boolean flag to indicate whether the parameter will remain at
    this value forever (`true`) or may change over time (`false`).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_4 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_5 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_operator_assign = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_validReplacement = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_accumulationFactor = R"doc()doc";

static const char *__doc_popart_Optimizer_clipNormSettings = R"doc()doc";

static const char *__doc_popart_Optimizer_clone = R"doc()doc";

static const char *__doc_popart_Optimizer_createOp = R"doc()doc";

static const char *__doc_popart_Optimizer_enableGradientAccumulation =
    R"doc()doc";

static const char *__doc_popart_Optimizer_enableReplicatedGraphs = R"doc()doc";

static const char *__doc_popart_Optimizer_factorsAreSetFromOptions =
    R"doc()doc";

static const char *__doc_popart_Optimizer_getAccumulationFactor = R"doc()doc";

static const char *__doc_popart_Optimizer_getClipNormSettings = R"doc()doc";

static const char *__doc_popart_Optimizer_getInputIds = R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingTensorId = R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingVal = R"doc()doc";

static const char *__doc_popart_Optimizer_getOptimizerInputs = R"doc()doc";

static const char *__doc_popart_Optimizer_getReplicatedGraphCount = R"doc()doc";

static const char *__doc_popart_Optimizer_gradientAccumulationEnabled =
    R"doc()doc";

static const char *__doc_popart_Optimizer_hash = R"doc()doc";

static const char *__doc_popart_Optimizer_lossScaling = R"doc()doc";

static const char *__doc_popart_Optimizer_ls = R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulation =
    R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulationEnabled =
    R"doc()doc";

static const char *__doc_popart_Optimizer_replicatedGraphCount = R"doc()doc";

static const char *__doc_popart_Optimizer_replicatedGraphsEnabled = R"doc()doc";

static const char *__doc_popart_Optimizer_resetTensorData = R"doc()doc";

static const char *__doc_popart_Optimizer_setFactorsFromOptions = R"doc()doc";

static const char *__doc_popart_Optimizer_setTensorData = R"doc()doc";

static const char *__doc_popart_Optimizer_type = R"doc()doc";

static const char *__doc_popart_Optimizer_type_s = R"doc()doc";

static const char *__doc_popart_Optimizer_validReplacement = R"doc()doc";

static const char *__doc_popart_PatternCreator = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator_2 = R"doc()doc";

static const char *__doc_popart_PatternNames = R"doc()doc";

static const char *__doc_popart_PatternNames_addName = R"doc()doc";

static const char *__doc_popart_PatternNames_contains = R"doc()doc";

static const char *__doc_popart_PatternNames_getInstance = R"doc()doc";

static const char *__doc_popart_PatternNames_getName = R"doc()doc";

static const char *__doc_popart_PatternNames_getName_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_names = R"doc()doc";

static const char *__doc_popart_Patterns = R"doc()doc";

static const char *__doc_popart_PatternsLevel = R"doc()doc";

static const char *__doc_popart_PatternsLevel_All = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Default = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Minimal = R"doc()doc";

static const char *__doc_popart_PatternsLevel_NoPatterns = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_2 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_3 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_4 = R"doc()doc";

static const char *__doc_popart_Patterns_create = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableCosGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableDecomposeBinaryConstScalar =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableDepthToSpaceOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpm1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableGemmDecomposition = R"doc()doc";

static const char *__doc_popart_Patterns_enableInPlace = R"doc()doc";

static const char *__doc_popart_Patterns_enableInitAccumulate = R"doc()doc";

static const char *__doc_popart_Patterns_enableLog1pGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableLogGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulLhsGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulRhsGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMulArgGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableNegativeOneScale = R"doc()doc";

static const char *__doc_popart_Patterns_enableNlllWithSoftMaxGradDirect =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableOpToIdentity = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_3 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_4 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePostNRepl = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enablePreUniRepl = R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomNormalLikeOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomUniformLikeOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableReciprocalGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableRuntimeAsserts = R"doc()doc";

static const char *__doc_popart_Patterns_enableSinGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableSoftMaxGradDirect = R"doc()doc";

static const char *__doc_popart_Patterns_enableSpaceToDepthOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableSplitGather = R"doc()doc";

static const char *__doc_popart_Patterns_enableSqrtGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableSubtractArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableUpdateInplacePrioritiesForIpu =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableUpsampleToResize = R"doc()doc";

static const char *__doc_popart_Patterns_enableZerosLikeOpPattern = R"doc()doc";

static const char
    *__doc_popart_Patterns_ensureAllMandatoryPreAliasPatternsAreEnabled =
        R"doc()doc";

static const char *__doc_popart_Patterns_getInplaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_getPreAliasList = R"doc()doc";

static const char *__doc_popart_Patterns_getRuntimeAssertsOn = R"doc()doc";

static const char *__doc_popart_Patterns_getSettings = R"doc()doc";

static const char *
    __doc_popart_Patterns_getUpdateInplacePrioritiesForIpuEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_inplaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isCosGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isDecomposeBinaryConstScalarEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isDepthToSpaceOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isExpGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isExpm1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isFmodArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isGemmDecompositionEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isInPlaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isInitAccumulateEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isLog1pGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isLogGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulLhsGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulRhsGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMulArgGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isNegativeOneScaleEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isNlllWithSoftMaxGradDirectEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isOpToIdentityEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_3 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_4 = R"doc()doc";

static const char *__doc_popart_Patterns_isPostNReplEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPreUniReplEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isRandomNormalLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isRandomUniformLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isReciprocalGradOpEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSinGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSoftMaxGradDirectEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSpaceToDepthOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSplitGatherEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSqrtGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSubtractArg1GradOpEnabled =
    R"doc()doc";

static const char
    *__doc_popart_Patterns_isUpdateInplacePrioritiesForIpuEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isUpsampleToResizeEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isZerosLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_operator_eq = R"doc()doc";

static const char *__doc_popart_Patterns_runtimeAssertsOn = R"doc()doc";

static const char *__doc_popart_Patterns_settings = R"doc()doc";

static const char *__doc_popart_Patterns_updateInplacePrioritiesForIpuEnabled =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo =
    R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_enabledByDefault =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_factory =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_mandatory =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_name = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_PreAliasPatternManager =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_createPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInfo = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInstance =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternInfos =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternName =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_opReplacementPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_patternInfos =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_patternTypeToTypeIndex =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_tryGetTypeIndex =
    R"doc()doc";

static const char *__doc_popart_RecomputationType =
    R"doc(Enum type to specify which ops to recompute in the backwards pass when
doing auto-recomputation.)doc";

static const char *__doc_popart_RecomputationType_N =
    R"doc(The number of RecomputationTypes values.)doc";

static const char *__doc_popart_RecomputationType_None =
    R"doc(No ops are recomputed.)doc";

static const char *__doc_popart_RecomputationType_NormOnly =
    R"doc(Only Norm ops (+ non-linearities, if following) are recomputed.)doc";

static const char *__doc_popart_RecomputationType_Pipeline =
    R"doc(Recompute all forward pipeline stages.)doc";

static const char *__doc_popart_RecomputationType_Standard =
    R"doc(Algorithm to pick checkpoints to try and minimise max liveness.)doc";

static const char *__doc_popart_RecomputeType = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_RemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_info = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_repeats = R"doc()doc";

static const char *__doc_popart_ReplicatedTensorSharding =
    R"doc(Enum type to specify whether to shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_Off =
    R"doc(Don't shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_On =
    R"doc(Do shard tensors over replicas.)doc";

static const char *__doc_popart_RequireOptimalSchedule = R"doc()doc";

static const char *__doc_popart_SGD =
    R"doc(Stochastic Gradient Descent (SGD) optimizer.

Akin to any optimizer implementation, this class is responsible for
updating each weight tensor ($w$) in the model using the gradient
($g$) of the loss function with respect to the weight as calculated
during the backwards pass.

The SGD optimizer has the following **state** for each weight:

* *velocity* ($v$)

The SGD optimizer has the following **hyper parameters**:

* *learning rate* ($\text{lr}$) * *momentum* ($\text{mm}$) * *weight
decay* ($\text{wd}$) * *dampening* ($\text{dm}$) * *velocity scaling*
($\text{vs}$) * *loss scaling* ($\text{ls}$) * *clip norm settings*

The values of these parameters can be shared between all weights but
some can be overridden with weight-specific values (see
SGD::insertSpecific). Hyper parameters are captured using
OptimizerValue objects and therefore can be either a constant value or
a non-constant value that can be adjusted by the user.

In the following we will describe how this optimizer updates a weight
using a gradient. In the context of this description the gradient is
is the value of the gradient *after* any gradient accumulation has
been performed and *after* the application of a loss scaling factor to
the gradient has been corrected for.

When the optimizer needs to update a weight, $w$, using a gradient,
$g$, it first updates the optimizer state as follows:

\f[ v' := v * \text{mm} + (1 - \text{dm}) * (g + \text{wd} * w) \text{
\ . } \f]

Following the update of the optimizer state the optimizer uses said
state to update the weight:

\f[ w' := w - \text{lr} * v' \text{ \ . } \f]

In addition to the above, the *velocity scaling* hyper parameter is a
scaling factor that can provide improved numerical stability by
ensuring the values stored in the optimizer state, $v$, are scaled by
this value. When using this parameter PopART will automatically deal
with the artificially scaled velocity value during the weight update
and other hyper parameters do not need to be adjusted).

In addition, the *loss scaling* hyper parameter is similar in nature
to the velocity scaling parameter. It is a scaling value that is
applied to the loss gradient at the start of the the backwards pass
and, at the end of the backwards pass, this scaling is reversed by
multiplying the gradients for each weight with the inverse of the loss
scaling value prior to updating the optimizer state. Using loss
scaling can also improve numerical stability in some cases.

Finally, it is possible to add clip norm settings for this optimizer.
These clip norms compute the L2 norm for a group of weights and adds a
scalar term to the weight update that effectively divides it by the
norm (or a constant value that is provided as part of the clip norm,
which ever is greater).)doc";

static const char *__doc_popart_SGD_SGD =
    R"doc(Constructor.

Parameter ``defaultLearningRate``:
    the learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    the weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultMomentum``:
    the momentum value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultDampening``:
    the dampening value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultVelocityScaling``:
    the velocity scaling value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``lossScaling``:
    the loss scaling value to use.

Parameter ``clipNormSettings``:
    a vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_SGD_SGD_2 =
    R"doc(Constructor.

Parameter ``params``:
    a parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"` and the map's values
    pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.

Parameter ``clipNormSettings``:
    a vector of ClipNormSettings (this can be used to set maximum
    values for weights).

**EXAMPLE**:

```
SGD({{"defaultLearningRate", {0.02, False}},
    {"defaultMomentum":{0.6, True}}});
```

This will create an SGD Optimizer which has a constant momentum of 0.6
and a changeable learning rate initially of 0.02. All OptimizerValues
not present in the map will take values from the `getUnset`*
functions.)doc";

static const char *__doc_popart_SGD_SGD_3 =
    R"doc(Construct an SDG instance with default values.)doc";

static const char *__doc_popart_SGD_SGD_4 = R"doc()doc";

static const char *__doc_popart_SGD_clone = R"doc()doc";

static const char *__doc_popart_SGD_createOp = R"doc()doc";

static const char *__doc_popart_SGD_dampenings = R"doc()doc";

static const char *__doc_popart_SGD_dps = R"doc()doc";

static const char *__doc_popart_SGD_dpsf1helper = R"doc()doc";

static const char *__doc_popart_SGD_fromDefaultMap = R"doc()doc";

static const char *__doc_popart_SGD_getComplete = R"doc()doc";

static const char *__doc_popart_SGD_getInputIds = R"doc()doc";

static const char *__doc_popart_SGD_getOptimizerInputs = R"doc()doc";

static const char *__doc_popart_SGD_getStoredValue = R"doc()doc";

static const char *__doc_popart_SGD_getUnsetDampening =
    R"doc(Default dampening value.)doc";

static const char *__doc_popart_SGD_getUnsetLearningRate =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_SGD_getUnsetLossScaling =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetMomentum =
    R"doc(Default momentum value.)doc";

static const char *__doc_popart_SGD_getUnsetVelocityScaling =
    R"doc(Default velocity scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetWeightDecay =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_SGD_hasSpecific = R"doc()doc";

static const char *__doc_popart_SGD_hash = R"doc()doc";

static const char *__doc_popart_SGD_insertSpecific =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    the TensorId of the weight.

Parameter ``learningRate``:
    the learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    the weight decay value to use for this specific weight.

Parameter ``momentum``:
    the momentum value to use for this specific weight.

Parameter ``dampening``:
    the dampening value to use for this specific weight.

Parameter ``velocityScaling``:
    the velocity scaling value to use for this specific weight.)doc";

static const char *__doc_popart_SGD_insertSpecific_2 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    the TensorId of the weight.

Parameter ``params``:
    a parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"` and the map's values
    pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_SGD_learningRates = R"doc()doc";

static const char *__doc_popart_SGD_lrs = R"doc()doc";

static const char *__doc_popart_SGD_mms = R"doc()doc";

static const char *__doc_popart_SGD_momentums = R"doc()doc";

static const char *__doc_popart_SGD_requiresAccl = R"doc()doc";

static const char *__doc_popart_SGD_resetTensorData = R"doc()doc";

static const char *__doc_popart_SGD_runValueChecks = R"doc()doc";

static const char *__doc_popart_SGD_setTensorData = R"doc()doc";

static const char *__doc_popart_SGD_slr0helper = R"doc()doc";

static const char *__doc_popart_SGD_slr1helper = R"doc()doc";

static const char *__doc_popart_SGD_smm1helper = R"doc()doc";

static const char *__doc_popart_SGD_swd1helper = R"doc()doc";

static const char *__doc_popart_SGD_type = R"doc()doc";

static const char *__doc_popart_SGD_type_s = R"doc()doc";

static const char *__doc_popart_SGD_validReplacement = R"doc()doc";

static const char *__doc_popart_SGD_velocityScalings = R"doc()doc";

static const char *__doc_popart_SGD_vss = R"doc()doc";

static const char *__doc_popart_SGD_wds = R"doc()doc";

static const char *__doc_popart_SGD_wdsf0helper = R"doc()doc";

static const char *__doc_popart_SGD_weightDecays = R"doc()doc";

static const char *__doc_popart_Session =
    R"doc(Session is a runtime instance the provides an interface for executing
ONNX graphs on IPU hardware.)doc";

static const char *__doc_popart_SessionOptions = R"doc()doc";

static const char *__doc_popart_SessionOptions_2 = R"doc()doc";

static const char *__doc_popart_SessionOptions_3 =
    R"doc(A structure containing user configuration options for the `Session`
class)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles =
    R"doc(A wrapper class for the `numIOTiles` option that permits any int value
and has an 'unassigned' state.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles_2 =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_assign =
    R"doc(Assign value using int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_eq =
    R"doc(Compare with ints.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_int =
    R"doc(Auto convert to int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_userAssignedValue =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_value = R"doc()doc";

static const char *__doc_popart_SessionOptions_accumulateOuterFragmentSettings =
    R"doc(Configuration setting for operations in the accumulate outer fragment.)doc";

static const char *__doc_popart_SessionOptions_accumulationFactor =
    R"doc(Specify the number of micro-batches to accumulate before applying the
varUpdate.)doc";

static const char *__doc_popart_SessionOptions_accumulationReductionType =
    R"doc(Specify how gradients are reduced when using gradient accumulation.
The options are equivilent to how gradients are reduced on lossOps.)doc";

static const char
    *__doc_popart_SessionOptions_accumulatorTensorLocationSettings =
        R"doc(Tensor location for gradient accumulator tensors.)doc";

static const char
    *__doc_popart_SessionOptions_activationTensorLocationSettings = R"doc()doc";

static const char *__doc_popart_SessionOptions_aliasZeroCopy =
    R"doc(Enable zero-copy for subgraphs)doc";

static const char *__doc_popart_SessionOptions_subgraphCopyingStrategy =
    R"doc(This setting determines how copies for inputs and outputs for
subgraphs are lowered. By setting this value to JustInTime you may
save memory at the cost of fragmenting subgraphs into multiple Poplar
functions.)doc";

static const char *__doc_popart_SessionOptions_autoRecomputation =
    R"doc(Enable recomputation of operations in the graph in the backwards pass
to reduce model size at the cost of computation cycles)doc";

static const char *__doc_popart_SessionOptions_batchSerializationSettings =
    R"doc(Configuration setting for batch serialization)doc";

static const char *__doc_popart_SessionOptions_cachePath =
    R"doc(Path to save the poplar::Executable to.)doc";

static const char *__doc_popart_SessionOptions_compileEngine =
    R"doc(when false, the backend will build the Poplar graph, but do not
compile it into an Engine. When this option is set, no execution can
be performed, and nothing can be transferred to the device. Functions
which retrieve information from the graph building stage will be ok
(tile mapping).)doc";

static const char *__doc_popart_SessionOptions_constantWeights =
    R"doc(An optimization for an inference session to have constant weights,
true by default. Set this option to false if you are going to want to
change the weights with a call to Session::resetHostWeights after the
session has been prepared. This option has no effect on a training
session)doc";

static const char *__doc_popart_SessionOptions_convolutionOptions =
    R"doc(Poplar convolution options)doc";

static const char *__doc_popart_SessionOptions_lstmOptions =
    R"doc(Poplar LSTM options)doc";

static const char *__doc_popart_SessionOptions_customCodeletCompileFlags =
    R"doc(Compile flags for the custom codelets. For example `-g` to generate
debug info.)doc";

static const char *__doc_popart_SessionOptions_customCodelets =
    R"doc(List of codelets (with filetype) to be added to the Poplar graph. See
the Poplar documentation for more information.)doc";

static const char *__doc_popart_SessionOptions_decomposeGradSum =
    R"doc(Replaces single sums of partial gradients with a tree of additions.
This can reduce max liveness at the cost of extra cycles. A typical
use case for this would be if a large weight tensor is used as an
input to many operations)doc";

static const char *__doc_popart_SessionOptions_delayVarUpdates =
    R"doc(Options to delay var updates as much as possible)doc";

static const char
    *__doc_popart_SessionOptions_disableGradAccumulationTensorStreams =
        R"doc(If true, the weight gradient tensors are not saved off the device when
devicex.weightsFromHost() is called. Note: this option is overridden
if syntheticDataMode is not Off.)doc";

static const char *__doc_popart_SessionOptions_dotChecks =
    R"doc(When to write '.dot' files during Ir construction)doc";

static const char *__doc_popart_SessionOptions_dotOpNames =
    R"doc(Include the Op name in the .dot file (the Op type is always exported))doc";

static const char *__doc_popart_SessionOptions_enableDistributedReplicatedGraphs =
    R"doc(Enable training with Poplar replicated graphs across multiple PopART
instances)doc";

static const char *__doc_popart_SessionOptions_enableEngineCaching =
    R"doc(Enable Poplar executable caching)doc";

static const char *__doc_popart_SessionOptions_enableFloatingPointChecks =
    R"doc(Throw an exception when floating point errors occur.)doc";

static const char *__doc_popart_SessionOptions_enableFullyConnectedPass =
    R"doc(Enable the global fullyConnectedPass option for matmuls)doc";

static const char *__doc_popart_SessionOptions_enableGradientAccumulation =
    R"doc(Enable gradient accumulation)doc";

static const char *__doc_popart_SessionOptions_enableGroupedMatmuls =
    R"doc(Enable/disable the grouping of matmuls that are the same shape)doc";

static const char *__doc_popart_SessionOptions_enableLoadAndOffloadRNGState =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_enableNonStableSoftmax =
    R"doc(By default, we use the stable-softmax Poplar function. This input
tensor to softmax, _x_, is preprocessed by subtracting max(_x_) to
each element before computing the exponentials, ensuring numerical
stability. If you are sure the inputs to your softmax operations are
small enough to not cause overflow when computing the exponential, you
can enable the non-stable version instead for speed-up)doc";

static const char *__doc_popart_SessionOptions_enableOutlining =
    R"doc(Identify and extract repeated parts of computational graph into
subgraphs.)doc";

static const char *__doc_popart_SessionOptions_enableOutliningCopyCostPruning =
    R"doc(When `true` the cost of copying of cached sections should be included
in the outlining cost model.)doc";

static const char *__doc_popart_SessionOptions_enablePipelining =
    R"doc(Enable pipelining of virtual graphs)doc";

static const char *__doc_popart_SessionOptions_enablePrefetchDatastreams =
    R"doc(By default, we will use prefetching for input data streams. Poplar
will speculative read data for a stream before is is required to allow
the 'preparation' of the data to occur in parallel with compute)doc";

static const char *__doc_popart_SessionOptions_enableReplicatedGraphs =
    R"doc(Enable replication of graphs)doc";

static const char *__doc_popart_SessionOptions_enableSerializedMatmuls =
    R"doc(Enable/disable the serializing of matmuls.)doc";

static const char *__doc_popart_SessionOptions_enableStableNorm =
    R"doc(If true, computes the mean first and subtracts the activations from it
before computing the variance. The implementation with this flag set
to true is slower than when set to false. The stable version requires
the first order moment to be estimated and applied to the sample set
before the second order central moment is calculated.)doc";

static const char *__doc_popart_SessionOptions_enableStochasticRounding =
    R"doc(Enable stochastic rounding)doc";

static const char *__doc_popart_SessionOptions_engineOptions =
    R"doc(Poplar engine options)doc";

static const char *__doc_popart_SessionOptions_executionPhaseSettings =
    R"doc(Configuration settings for execution phases)doc";

static const char *__doc_popart_SessionOptions_explicitRecomputation =
    R"doc(Enable explicit recomputation)doc";

static const char *__doc_popart_SessionOptions_exportPoplarComputationGraph =
    R"doc(Export Poplar computation graph)doc";

static const char *__doc_popart_SessionOptions_exportPoplarVertexGraph =
    R"doc(Export Poplar vertex graph)doc";

static const char *__doc_popart_SessionOptions_finalDotOp =
    R"doc(See #firstDotOp.)doc";

static const char *__doc_popart_SessionOptions_firstDotOp =
    R"doc(The ops to write to the .dot file will be a continuous interval of the
schedule, controlled by firstDotOp and finalDotOp. In particular, it
will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp)))doc";

static const char *__doc_popart_SessionOptions_gclOptions =
    R"doc(GCL options)doc";

static const char *__doc_popart_SessionOptions_getPrefetchBufferingDepth =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_globalReplicaOffset =
    R"doc(The first replica index that this PopART instance is running)doc";

static const char *__doc_popart_SessionOptions_globalReplicationFactor =
    R"doc(The total number of replicas in a multi instance replicated graph
training session (this should be left as the default value (1) if
distributed replicated graphs are disabled))doc";

static const char *__doc_popart_SessionOptions_groupHostSync =
    R"doc(Allows to group the streams from host at the beginning and the streams
to host at the end, this trades off sum-liveness efficiency for cycle
efficiency.)doc";

static const char *__doc_popart_SessionOptions_hardwareInstrumentations =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_hostAllReduce =
    R"doc(Perform AllReduce operation on the host. Only useful for training
session)doc";

static const char *__doc_popart_SessionOptions_hostAllReduceRemoteBuffer =
    R"doc(Enable the use of poplar::RemoteBuffers for hostAllReduce operations)doc";

static const char *__doc_popart_SessionOptions_hostWeightUpdate =
    R"doc(Perform weight update on the host. Only useful for training session)doc";

static const char
    *__doc_popart_SessionOptions_instrumentWithHardwareCycleCounter =
        R"doc(Add instrumentation to your program to count the number of device
cycles (a single tile, on a single IPU) that your main program takes
to execute. Expect this to have a small detrimental impact on
performance.)doc";

static const char *__doc_popart_SessionOptions_kahnTieBreaker =
    R"doc(The initial scheduling is done with Kahn's algorithm. When several Ops
are free to be scheduled, this controls which method is used)doc";

static const char *__doc_popart_SessionOptions_logDir =
    R"doc(A directory for log traces to be written into)doc";

static const char *__doc_popart_SessionOptions_looseThresholdAtPeak =
    R"doc(The AutoLoose VarUpudate merging algorithm has absolute threshold
defined by min(mergeVarUpdateMemThreshold, liveAtPeak - liveCurrently
+ looseThresholdAtPeak), where liveAtPeak is an estimate of the
maximum live memory of the computation, and liveCurrently is an
estimate of the live memory where the threshold is being used to
determine whether to schedule or postpone a VarUpdate.)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdate =
    R"doc(Enable merging of VarUpdates into groups of VarUpdates, by flattening
and concatenating Variable Tensors and Updating Tensors)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdateMemThreshold =
    R"doc(The AutoLoose and AutoTight VarUpdate merging algorithm has a
threshold on the total memory of Variable Tensors to merge for
updating. Memory in bytes.)doc";

static const char *__doc_popart_SessionOptions_numIOTiles =
    R"doc(Number of IPU tiles dedicated to IO.)doc";

static const char *__doc_popart_SessionOptions_operator_assign = R"doc()doc";

static const char
    *__doc_popart_SessionOptions_optimizerStateTensorLocationSettings =
        R"doc(Tensor location for optimizer state tensors.)doc";

static const char *__doc_popart_SessionOptions_opxAliasChecking =
    R"doc(Run Opx checks to verify IR tensor aliasing information corresponds to
lowered Poplar tensor aliasing)doc";

static const char *__doc_popart_SessionOptions_opxModifyChecking =
    R"doc(Run Opx checks to verify IR tensor modification information
corresponds to lowered Poplar tensor modifications)doc";

static const char *__doc_popart_SessionOptions_outlineSequenceBreakCost =
    R"doc(The penalty applied to outlining potential sub-graphs if the sub-graph
to be created breaks up a sequence of operations that are more
efficient (for example for overlapping compute and exchange) when
outlined together Default value is set to ~10 *
Op::getHighSubgraphValue().)doc";

static const char *__doc_popart_SessionOptions_outlineThreshold =
    R"doc(The incremental value that a sub-graph requires, relative to its
nested sub-graphs (if any), to be eligible for outlining. A high
threshold results in fewer sub-graphs being outlined, a negative value
results in all being outlined. The gross value of a sub-graph is the
sum of its constituent Ops' Op::getSubgraphValue() values. To disable
outlining, it is better to set enableOutlining to false than to set
this value to infinity. The default value of 1.0f results in all high
value operations such as convolution being cached, but standalone low
Value operations such as Relu will not be.)doc";

static const char *__doc_popart_SessionOptions_partialsTypeMatMuls =
    R"doc(Set the partials type globally for matmuls. Can be overridden
individually with `builder.setPartialsType()`. Valid values are
`"float"` and `"half"`. By default, this is not set, so no global
partials type is imposed.)doc";

static const char *__doc_popart_SessionOptions_prefetchBufferingDepthMap =
    R"doc(When #enablePrefetchDatastreams is set this mapping can be used to set
tensor-specific buffering depths for tensors that are streamed to the
host (typically input tensors). This buffering depth could be
envisaged as being the size of a circular buffer that feeds data to
Poplar. A buffering depth greater than `1` may improve the performance
due to increased parallelisation but comes at the cost of increasing
the memory footprint. Streams for tensors that have no entry in this
map default to a buffering depth of `1`.)doc";

static const char *__doc_popart_SessionOptions_rearrangeAnchorsOnHost =
    R"doc(Before anchor tensors are streamed from device to host, they are not
necessarily arranged in memory as required when they are to be copied
from host stream to host. This can be done on the device or on the
host. Done on host by default to save memory, but often at the expense
of cycles, especially for larger anchor tensors.)doc";

static const char *__doc_popart_SessionOptions_replicatedGraphCount =
    R"doc(If enableReplicatedGraphs is true, replicatedGraphCount will set the
number of model replications. E.g. if your model uses 1 IPU, a
replicatedGraphCount of 2 will use 2 IPUs. If your model is pipelined
across 4 IPUs, a replicatedGraphCount of 4 will use 16 IPUs total.
Therefore the number of IPUs you request must be a multiple of
replicatedGraphCount. If the training is done across multiple
instances then the replicatedGraphCount is the number of replicas for
this instance.)doc";

static const char *__doc_popart_SessionOptions_reportOptions =
    R"doc(Poplar reporting options)doc";

static const char *__doc_popart_SessionOptions_separateCallOpPdfs =
    R"doc(When generating PDFs of IR graphs, create separate PDFs for each
subgraph.)doc";

static const char
    *__doc_popart_SessionOptions_serializedPoprithmsAnnealGraphsDir =
        R"doc(PopART uses Poprithms for scheduling PopART Graphs. The Poprithms
Graphs created for scheduling can be optionally serialised (written to
file). The string below specified the directory to serialize Poprithms
Graphs to. If it is empty, then the Graphs will not be serialised. The
names of serialization files will be poprithms_anneal_graph_`i'.json
for the lowest non-existing `i's. The directory must already exist,
PopART will not create it.)doc";

static const char *__doc_popart_SessionOptions_strictOpVersions =
    R"doc(Strict op version checks will throw an error if the exact version of
an op required for the models opset is not supported. Turning this
check off will cause PopART to fall back to the latest implementation
of the op that is supported. Warning, turning off these checks may
cause undefined behaviour.)doc";

static const char *__doc_popart_SessionOptions_swapLimitScheduler =
    R"doc(The maximum number of improving steps allowed by the scheduling
algorithm before a solution must be returned)doc";

static const char *__doc_popart_SessionOptions_syntheticDataMode =
    R"doc(Use synthetic data i.e. disable data transfer to/from the host Set to
'Off' to use real data)doc";

static const char *__doc_popart_SessionOptions_tensorLocationSettingsOverride =
    R"doc(Override tensor location for specific tensors by setting a
TensorLocation for specific TensorId values.)doc";

static const char *__doc_popart_SessionOptions_timeLimitScheduler =
    R"doc(The maximum allowed time that can be spent searching for a good Graph
schedule before a solution must be returned)doc";

static const char *__doc_popart_SessionOptions_virtualGraphMode =
    R"doc(This option allows you to place ops on virtual graphs to achieve model
parallelism - either manually using model annotations, or
automatically)doc";

static const char *__doc_popart_SessionOptions_weightTensorLocationSettings =
    R"doc(Tensor location for weight tensors.)doc";

static const char *__doc_popart_Session_Session = R"doc()doc";

static const char *__doc_popart_Session_assertExecutableLoaded =
    R"doc(Throws an error if there is no executable.)doc";

static const char *__doc_popart_Session_compileAndExport =
    R"doc(Compiles the graph and exports it to the specified path

This will create a poplar::Graph and compile the poplar::Executable
before exporting the executable and metadata to allow offline running.

\arg executablePath path to output the compiled executable and
associated metadata: if empty, these will not be exported \arg
weightsPath path to output the weights: if empty, these will not be
exported)doc";

static const char *__doc_popart_Session_device =
    R"doc(Implementation of the computation, for IPU back-end this is where
calls to poplar are made.)doc";

static const char *__doc_popart_Session_deviceInfo =
    R"doc(Information about the device which this session uses)doc";

static const char *__doc_popart_Session_executable =
    R"doc(The final executable which contains all the data, metadata and
configuration parameters necessary to start running the program on the
device.)doc";

static const char *__doc_popart_Session_exportInputs =
    R"doc(Export numElements from stepIO.in)doc";

static const char *__doc_popart_Session_getCycleCount =
    R"doc(Copy the cycle count tensor to host from the device)doc";

static const char *__doc_popart_Session_getDevice = R"doc()doc";

static const char *__doc_popart_Session_getExecutable = R"doc()doc";

static const char *__doc_popart_Session_getExecutionReport =
    R"doc(Retrieve the execution report from the poplar::Engine

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the `prepareDevice()` call has been
made.

\arg useCbor Produce a CBOR formatted report \arg resetProfile Resets
the execution profile

Returns:
    a string containing the execution report)doc";

static const char *__doc_popart_Session_getGraphReport =
    R"doc(Retrieve the graph report from the poplar::Engine

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the `prepareDevice()` call has been
made.

\arg useCbor Produce a CBOR formatted report

Returns:
    a string containing the graph (compilation) report)doc";

static const char *__doc_popart_Session_getInfo =
    R"doc(get the TensorInfo on a Tensor)doc";

static const char *__doc_popart_Session_getIr = R"doc()doc";

static const char *__doc_popart_Session_getIrLowering = R"doc()doc";

static const char *__doc_popart_Session_getRNGState = R"doc()doc";

static const char *__doc_popart_Session_getSerializedGraph =
    R"doc(Retrieve the serialized graph from the poplar::Engine

A JSON format report is produced.

This may only be called after the `prepareDevice()` call has been
made.

Returns:
    a string containing the serialized graph)doc";

static const char *__doc_popart_Session_getSummaryReport =
    R"doc(Retrieve the summary from from the poplar::Engine

The options which were given to the constructor will influence the
information in the report.

This may only be called after the `prepareDevice()` call has been
made. \arg resetProfile Resets the execution profile

Returns:
    a string containing the report)doc";

static const char *__doc_popart_Session_getTensorTileMap =
    R"doc(Retrieve the tensor tile mapping from the poplar::Graph

This may only be called after the `prepareDevice()` call has been
made.

Returns:
    a TensorTileMap object for all tensors in the graph)doc";

static const char *__doc_popart_Session_ir =
    R"doc(abstraction of the computation, the Ir is where all the compute graph
optimisations, backwards pass construction, re-computation growing
etc. happens.)doc";

static const char *__doc_popart_Session_loadExecutableFromFile =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
file. The file must have been created with compileAndExport()

Parameter ``filename``:
    Name of the file to load the executable from.)doc";

static const char *__doc_popart_Session_loadExecutableFromStream =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
stream. The stream must have been created with compileAndExport()

Parameter ``in``:
    Stream to load the executable from.)doc";

static const char *__doc_popart_Session_lowering =
    R"doc(Implementation of the lowering of the PopART Ir to the poplar Graph.)doc";

static const char *__doc_popart_Session_modelToHost =
    R"doc(Write current model to ONNX file)doc";

static const char *__doc_popart_Session_prepareDevice =
    R"doc(Prepare the network for execution.

This will create the poplar::Graph, poplar::Engine, and setting up
poplar::Streams.)doc";

static const char *__doc_popart_Session_readWeights =
    R"doc(Read the weights. Must have called weightsToHost first

weight data : to addresses in weightsIo.out)doc";

static const char *__doc_popart_Session_resetHostWeights =
    R"doc(Reset the weights with the weights in a ONNX model that differs to the
current model only in weights. This only updates the weights on the
host; the user still needs to call weightsFromHost() after this to
update the weights on the device.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf

Parameter ``ignoreWeightsInModelWithoutCorrespondingHostWeight``:
    If true, do not error if there are initializers in the ONNX model
    with no corresponding initializer tensor in the session's IR)doc";

static const char *__doc_popart_Session_run =
    R"doc(Perform one step.

input data : from address in stepIO.in debug name : debug string to
identify this run in logs output data : to addresses in stepIO.out)doc";

static const char *__doc_popart_Session_runCalled =
    R"doc(Flag to indicate if run has been called)doc";

static const char *__doc_popart_Session_serializeIr =
    R"doc(Serizalise the ir graph to a string

format : the format to serialize)doc";

static const char *__doc_popart_Session_setDevice =
    R"doc(Select a device type.

/param deviceInfo which defines the type of device to work on)doc";

static const char *__doc_popart_Session_setRNGState = R"doc()doc";

static const char *__doc_popart_Session_setRandomSeed = R"doc()doc";

static const char *__doc_popart_Session_tryLoadExecutable =
    R"doc(Attempts to load a serialized executable. If succesful then Ir
preparation and `poplar::Graph` compilation are skipped.)doc";

static const char *__doc_popart_Session_updateExternallySavedTensorLocations =
    R"doc(Update the tensor locations of the tensors in the Session's ONNX
model. The new file will be created at this point, and written to when
the ONNX model is saved with a subsequent call to modelToHost.

Parameter ``fromLocation``:
    All externally saved tensors with location fromLocation will have
    their location updated to toLocation.

Parameter ``toLocation``:
    The updated location. Must not already exist.)doc";

static const char *__doc_popart_Session_weightsFromHost =
    R"doc(write to device, from an ONNX model loaded from directory Currently,
the weights are taken from the onnx Model passed to the constructor,
but this should be relaxed so that the weights can come from any Model)doc";

static const char *__doc_popart_Session_weightsFromHostCalled =
    R"doc(Flag to indicate if weightsFromHost has been called)doc";

static const char *__doc_popart_Session_weightsToHost =
    R"doc(Copy the weights to host from the device)doc";

static const char *__doc_popart_Session_writeWeights =
    R"doc(Write the weights. Must call weightsFromHost after

weight data : to addresses in weightsIo.out)doc";

static const char *__doc_popart_StepIOGeneric = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_array = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_offset = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_StepIOGeneric = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_advance = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_assertNumElements = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_get = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_getTensorInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_in = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inComplete = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inputsInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_out = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_outputsInfo = R"doc()doc";

static const char *__doc_popart_StepIOSplitter = R"doc()doc";

static const char *__doc_popart_SyncPattern = R"doc()doc";

static const char *__doc_popart_SyncPattern_Full = R"doc()doc";

static const char *__doc_popart_SyncPattern_ReplicaAndLadder = R"doc()doc";

static const char *__doc_popart_SyncPattern_SinglePipeline = R"doc()doc";

static const char *__doc_popart_SyntheticDataMode =
    R"doc(Enum type used to specify the data source for input tensors.)doc";

static const char *__doc_popart_SyntheticDataMode_N =
    R"doc(The number of SyntheticDataMode values.)doc";

static const char *__doc_popart_SyntheticDataMode_Off =
    R"doc(Use real data.)doc";

static const char *__doc_popart_SyntheticDataMode_RandomNormal =
    R"doc(Input tensors are initialised with distribution ~N(0,1).)doc";

static const char *__doc_popart_SyntheticDataMode_Zeros =
    R"doc(Input tensors are initialised to all zeros.)doc";

static const char *__doc_popart_TensorData = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData_2 = R"doc()doc";

static const char *__doc_popart_TensorData_copyDataAs = R"doc()doc";

static const char *__doc_popart_TensorData_data = R"doc()doc";

static const char *__doc_popart_TensorData_data_2 = R"doc()doc";

static const char *__doc_popart_TensorData_data_3 = R"doc()doc";

static const char *__doc_popart_TensorData_resetData = R"doc()doc";

static const char *__doc_popart_TensorData_resetData_2 = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_complete = R"doc()doc";

static const char
    *__doc_popart_TensorGradRegistry_decrementNumberExpectedEdges = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_expectedNumEdges =
    R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_getNumberExpectedEdges =
    R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_insert = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_partial = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_popComplete = R"doc()doc";

static const char *__doc_popart_TensorGradRegistry_tryMakeComplete =
    R"doc()doc";

static const char *__doc_popart_TensorLocation =
    R"doc(Class that describes the memory characteristics of one or multiple
tensors.

See also: SessionOptions.)doc";

static const char *__doc_popart_TensorLocationSettings =
    R"doc(A structure containing user configuration for cache/offloading
settings.)doc";

static const char *__doc_popart_TensorLocationSettings_TensorLocationSettings =
    R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_2 = R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_3 = R"doc()doc";

static const char *__doc_popart_TensorLocationSettings_location =
    R"doc(The default tensor location for this tensor type.)doc";

static const char *__doc_popart_TensorLocationSettings_minElementsForOffChip =
    R"doc(A minimum number of elements below which offloading won't be
considered.)doc";

static const char *
    __doc_popart_TensorLocationSettings_minElementsForReplicatedTensorSharding =
        R"doc(A minimum number of elements below which Replicated Tensor Sharding
(RTS) won't be considered.)doc";

static const char *__doc_popart_TensorLocationSettings_operator_assign =
    R"doc()doc";

static const char *__doc_popart_TensorLocation_TensorLocation =
    R"doc(Equivalent to calling TensorLocation(TensorStorage::Undefined,
TileSet::Compute, TileSet::Compute, ReplicatedTensorSharding::Off))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_2 =
    R"doc(Equivalent to calling TensorLocation(storage, TileSet::Compute,
TileSet::Compute, ReplicatedTensorSharding::Off))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_3 =
    R"doc(Equivalent to calling TensorLocation(storage, TileSet::Compute,
TileSet::Compute, replicatedTensorSharding))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_4 =
    R"doc(Construct a TensorLocation from parameters.

Parameter ``storage``:
    the memory location of the tensor(s).

Parameter ``loadTileSet``:
    the tiles through which the tensor(s) are loaded onto the chip.

Parameter ``storageTileSet``:
    the tiles on which the tensor(s) are stored.

Parameter ``replicatedTensorSharding``:
    whether to apply replicated tensor sharding.)doc";

static const char *__doc_popart_TensorLocation_TensorLocation_5 = R"doc()doc";

static const char *__doc_popart_TensorLocation_isRemote = R"doc()doc";

static const char *__doc_popart_TensorLocation_loadTileSet =
    R"doc(The tiles through which the tensor(s) are loaded onto the chip.)doc";

static const char *__doc_popart_TensorLocation_operator_assign = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_eq = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_ne = R"doc()doc";

static const char *__doc_popart_TensorLocation_replicatedTensorSharding =
    R"doc(Whether to apply replicated tensor sharding (RTS) or not.)doc";

static const char *__doc_popart_TensorLocation_serialize = R"doc()doc";

static const char *__doc_popart_TensorLocation_storage =
    R"doc(The memory location of the tensor(s).)doc";

static const char *__doc_popart_TensorLocation_storageTileSet =
    R"doc(The tiles on which the tensor(s) are stored.)doc";

static const char *__doc_popart_TensorStorage =
    R"doc(Enum type that determines where a tensor is stored.)doc";

static const char *__doc_popart_TensorStorage_OffChip =
    R"doc(Store the tensor in streaming memory.)doc";

static const char *__doc_popart_TensorStorage_OnChip =
    R"doc(Store the tensor in on-chip memory.)doc";

static const char *__doc_popart_TensorStorage_Undefined =
    R"doc(Location unspecified.)doc";

static const char *__doc_popart_Tensors = R"doc()doc";

static const char *__doc_popart_Tensors_M = R"doc()doc";

static const char *__doc_popart_Tensors_Tensors = R"doc()doc";

static const char *__doc_popart_Tensors_addActGrad = R"doc()doc";

static const char *__doc_popart_Tensors_addConstInit = R"doc()doc";

static const char *__doc_popart_Tensors_addConstInit_2 = R"doc()doc";

static const char *__doc_popart_Tensors_addInit = R"doc()doc";

static const char *__doc_popart_Tensors_addStream = R"doc()doc";

static const char *__doc_popart_Tensors_addVarInit = R"doc()doc";

static const char *__doc_popart_Tensors_addVarInit_2 = R"doc()doc";

static const char *__doc_popart_Tensors_aliasChainsFrom = R"doc()doc";

static const char *__doc_popart_Tensors_aliasChainsTo = R"doc()doc";

static const char *__doc_popart_Tensors_aliases = R"doc()doc";

static const char *__doc_popart_Tensors_append = R"doc()doc";

static const char *__doc_popart_Tensors_clearAliases = R"doc()doc";

static const char *__doc_popart_Tensors_constIds = R"doc()doc";

static const char *__doc_popart_Tensors_contains = R"doc()doc";

static const char *__doc_popart_Tensors_contains_2 = R"doc()doc";

static const char *__doc_popart_Tensors_find = R"doc()doc";

static const char *__doc_popart_Tensors_get = R"doc()doc";

static const char *__doc_popart_Tensors_getAliasRegions = R"doc()doc";

static const char *__doc_popart_Tensors_getAliases = R"doc()doc";

static const char *__doc_popart_Tensors_getAllTensorIds = R"doc()doc";

static const char *__doc_popart_Tensors_getChainsFromTo = R"doc()doc";

static const char *__doc_popart_Tensors_getConstIds = R"doc()doc";

static const char *__doc_popart_Tensors_getIds = R"doc()doc";

static const char *__doc_popart_Tensors_getNoProducerIds = R"doc()doc";

static const char *__doc_popart_Tensors_getOfType = R"doc()doc";

static const char *__doc_popart_Tensors_graph = R"doc()doc";

static const char *__doc_popart_Tensors_insert = R"doc()doc";

static const char *__doc_popart_Tensors_insertConstId = R"doc()doc";

static const char *__doc_popart_Tensors_makeConstInit = R"doc()doc";

static const char *__doc_popart_Tensors_moveIntoTensors = R"doc()doc";

static const char *__doc_popart_Tensors_remove = R"doc()doc";

static const char *__doc_popart_Tensors_removeIsolated = R"doc()doc";

static const char *__doc_popart_Tensors_updateAliases = R"doc()doc";

static const char *__doc_popart_TileSet =
    R"doc(Enum type to specify a set of tiles.)doc";

static const char *__doc_popart_TileSet_Compute =
    R"doc(The set of tiles designated for compute operations.)doc";

static const char *__doc_popart_TileSet_IO =
    R"doc(The set of tiles designated for IO operations.)doc";

static const char *__doc_popart_TrainingSession = R"doc()doc";

static const char *__doc_popart_TrainingSession_TrainingSession = R"doc()doc";

static const char *__doc_popart_TrainingSession_configureFromOnnx = R"doc()doc";

static const char *__doc_popart_TrainingSession_connectStreamToCallback =
    R"doc(Connect Poplar stream callbacks. In conjunction with
`getGradAndVarStreamIds` the streams can be used to copy gradients to
the host to perform collective operations after which the variables
can be streamed back after they have been updated to the device.
`index` referes to the replica index when using replicated graphs.)doc";

static const char *__doc_popart_TrainingSession_copyFromRemoteBuffer =
    R"doc(Read from a RemoteBuffer object into a user space pointer w. This can
be useful when we run larger models with host side reductions since
HEXOPT is currently limited to 128 MB)doc";

static const char *__doc_popart_TrainingSession_copyToRemoteBuffer =
    R"doc(Write to a RemoteBuffer object from a user space pointer w. This can
be useful when we run larger models with host side reductions since
HEXOPT is currently limited to 128 MB)doc";

static const char *__doc_popart_TrainingSession_createFromOnnxModel =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for training

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches

Parameter ``loss``:
    The TensorId of the final scalar loss tensor for training

Parameter ``optimizer``:
    The name of an optimizer to use when training

Parameter ``userOptions``:
    String to configure session options

Parameter ``patterns``:
    Optimization patterns to apply)doc";

static const char *__doc_popart_TrainingSession_getHostReduceRemoteBuffers =
    R"doc(Access the remote buffers associated with gradient and weight streams
that are used in host side all reduce operations. Only populated if
hostAllReduce and hostAllReduceRemoteBuffer are enabled.)doc";

static const char *__doc_popart_TrainingSession_getHostReduceStreamIds =
    R"doc(Access the stream IDs for variables that are involved in host side
reductions on the host. Only populated if hostAllReduce is enabled in
the SessionOptions)doc";

static const char *__doc_popart_TrainingSession_updateOptimizerFromHost =
    R"doc(Update the optimizer and the associated hyperparameters but not the
optimizer state tensors.

**NOTE**: The optimizer parameter has to be compatible with the
optimizer passed to the constructor. For example, you cannot call this
function with an SDG1 optimizer if you created the session with an
SDG0 optimizer. The reason for this is that it is not possible to
change the IR after it has been constructed.

Parameter ``optimizer``:
    A pointer to a popart::Optimizer.)doc";

static const char *__doc_popart_VirtualGraphMode =
    R"doc(Enum type used to specify a virtual graph mode.)doc";

static const char *__doc_popart_VirtualGraphMode_Auto =
    R"doc(Use `autoVirtualGraph` transform.)doc";

static const char *__doc_popart_VirtualGraphMode_ExecutionPhases =
    R"doc(Virtual graphs are tied to execution phases.)doc";

static const char *__doc_popart_VirtualGraphMode_Manual =
    R"doc(User must set the `virtualGraph` attribute on all ops.)doc";

static const char *__doc_popart_VirtualGraphMode_N =
    R"doc(The number of VirtualGraphModes values.)doc";

static const char *__doc_popart_VirtualGraphMode_Off =
    R"doc(Virtual graphs are not enabled.)doc";

static const char *__doc_popart_WeightDecayMode =
    R"doc(Enum type for different types of weight decay.)doc";

static const char *__doc_popart_WeightDecayMode_Decay =
    R"doc(Weight decay (e.g. AdamW))doc";

static const char *__doc_popart_WeightDecayMode_L2Regularization =
    R"doc(L2 regularization (e.g. PyTorch-like Adam))doc";

static const char *__doc_popart_anchorSumPrefix = R"doc()doc";

static const char *__doc_popart_createRecomputedTensorId = R"doc()doc";

static const char *__doc_popart_cycleCountPrefix = R"doc()doc";

static const char *__doc_popart_dotCheckFromString = R"doc()doc";

static const char *__doc_popart_error = R"doc(Exception class for popart)doc";

static const char *__doc_popart_error_empty = R"doc()doc";

static const char *__doc_popart_error_error = R"doc()doc";

static const char *__doc_popart_error_error_2 = R"doc()doc";

static const char *__doc_popart_error_error_3 =
    R"doc(Variadic constructor for error which allows the user to use a fmt
string for the message.

throw error("This is an error reason {}", 42);)doc";

static const char *__doc_popart_error_error_4 = R"doc()doc";

static const char *__doc_popart_error_formatMessage =
    R"doc(As the fmt::format function can throw an exception itself we catch the
FormatError exception here and convert it to a popart exception.)doc";

static const char *__doc_popart_error_logMessage =
    R"doc(Log the exception message)doc";

static const char *__doc_popart_getDotCheckString = R"doc()doc";

static const char *__doc_popart_getEdgeGradId = R"doc()doc";

static const char *__doc_popart_getErrorSource = R"doc()doc";

static const char *__doc_popart_getGradId = R"doc()doc";

static const char *__doc_popart_getNonGradId = R"doc()doc";

static const char *__doc_popart_getNonRemoteArgTensorId = R"doc()doc";

static const char *__doc_popart_getOptMap = R"doc()doc";

static const char *__doc_popart_getRecompId = R"doc()doc";

static const char *__doc_popart_getRemoteArgTensorId = R"doc()doc";

static const char *__doc_popart_getUpdatedVarId = R"doc()doc";

static const char *__doc_popart_hash_value = R"doc()doc";

static const char *__doc_popart_hash_value_2 = R"doc()doc";

static const char *__doc_popart_hash_value_3 = R"doc()doc";

static const char *__doc_popart_hostReduceGradCopyPrefix = R"doc()doc";

static const char *__doc_popart_hostReduceVarCopyPrefix = R"doc()doc";

static const char *__doc_popart_internal_error =
    R"doc(Exception class specific to internal errors This should be used as an
assert; for states where the user should not have been able to create.)doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter = R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_CorrectnessAsserter =
        R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_aFact =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_bps =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkIn =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkOut =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_exe =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getArtDivisor =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getBaseError =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getInExpected =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getNElms =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getOutExpected =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_ir =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_onnxIns =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_rFact =
    R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadInputSize =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadOutputSize =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwIncorrectInput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingInput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingOutput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_warnOfUnunsedInput =
        R"doc()doc";

static const char *__doc_popart_iosizecheck_assertInCorrect = R"doc()doc";

static const char *__doc_popart_iosizecheck_assertOutCorrect = R"doc()doc";

static const char *__doc_popart_isGradId = R"doc()doc";

static const char *__doc_popart_isValidTensorLocation = R"doc()doc";

static const char *__doc_popart_memory_allocation_err = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_clone = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getGraphReport =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getSummaryReport =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_memory_allocation_err =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_NumericsReport =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_fullReport =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_getRelativeErrors =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_relerrs = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_report = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_reports = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker = R"doc()doc";

static const char
    *__doc_popart_numerics_NumericsTracker_calculateRelativeError = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_getRelativeError =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_insert = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dA = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dAB = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dB = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_str = R"doc()doc";

static const char *__doc_popart_operator_lshift = R"doc()doc";

static const char *__doc_popart_operator_lshift_2 = R"doc()doc";

static const char *__doc_popart_operator_lshift_3 =
    R"doc(Write a representation of a DeviceType to an output stream.

Parameter ``os``:
    Output stream

Parameter ``dt``:
    Device type reference

Returns:
    The same output stream for chaining)doc";

static const char *__doc_popart_operator_lshift_4 =
    R"doc(Write a representation of a DeviceConnectionType to an output stream.

Parameter ``os``:
    Output stream

Parameter ``dct``:
    Device connection type reference

Returns:
    The same output stream for chaining)doc";

static const char *__doc_popart_operator_lshift_5 =
    R"doc(Write a representation of a SyncPattern to an output stream.

Parameter ``os``:
    Output stream

Parameter ``sp``:
    Sync pattern reference

Returns:
    The same output stream for chaining)doc";

static const char *__doc_popart_operator_lshift_6 = R"doc()doc";

static const char *__doc_popart_operator_lshift_7 = R"doc()doc";

static const char *__doc_popart_operator_lshift_8 = R"doc()doc";

static const char *__doc_popart_operator_lshift_9 = R"doc()doc";

static const char *__doc_popart_operator_lshift_10 = R"doc()doc";

static const char *__doc_popart_operator_lshift_11 = R"doc()doc";

static const char *__doc_popart_operator_lshift_12 = R"doc()doc";

static const char *__doc_popart_operator_lshift_13 = R"doc()doc";

static const char *__doc_popart_popx_Devicex = R"doc()doc";

static const char *__doc_popart_popx_Devicex_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_Datastream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_getTensorId =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_io = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_setStepIO = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_streamId = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_tensor = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Devicex = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_InputDatastream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_read = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readComplete =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readPrefetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_OutputDatastream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_write =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_PrefetchCallback =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_complete =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_ds = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_fetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_prefetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostFromHostStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostToHostStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_chBuffers = R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRandomSeedStream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRngStateStream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectStreamToCallback =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_convCache = R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyFromRemoteBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyToRemoteBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCount = R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCountTensorToHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_d2hWeightBuffers = R"doc()doc";

static const char *__doc_popart_popx_Devicex_deviceInfo = R"doc()doc";

static const char *__doc_popart_popx_Devicex_doProfileChecks = R"doc()doc";

static const char *__doc_popart_popx_Devicex_engineIsLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_executable = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getAccumulationFactor =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getDeviceInfo = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getEfficientlyCreatedInputTensors =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getExecutionReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicaOffset =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicationFactor =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGraphReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceRemoteBuffers =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceStreamIds =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getLinearlyCreatedInputTensors =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getReplicationFactor = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getRngStateToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSerializedGraph = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSummaryReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getTensorTileMap = R"doc()doc";

static const char *__doc_popart_popx_Devicex_hostStreamToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_inputStreams = R"doc()doc";

static const char *__doc_popart_popx_Devicex_ir = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isEngineLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isReplicatedGraph = R"doc()doc";

static const char *__doc_popart_popx_Devicex_loadEngineAndConnectStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering = R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_matmulCache = R"doc()doc";

static const char *__doc_popart_popx_Devicex_nCallsToRun = R"doc()doc";

static const char *__doc_popart_popx_Devicex_optimizerFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_outputStreams = R"doc()doc";

static const char *__doc_popart_popx_Devicex_pEngine = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanConvolutions = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanMatMuls = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepare = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_readWeights = R"doc()doc";

static const char *__doc_popart_popx_Devicex_reconnectInputStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsFromHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsToHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_rngBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_saveTensorTileMap = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setEngineIsLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRandomSeedFromHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateValue = R"doc()doc";

static const char *__doc_popart_popx_Devicex_stepIoSplitter = R"doc()doc";

static const char *__doc_popart_popx_Devicex_trySaveTensorTileMap = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_writeWeights = R"doc()doc";

static const char *__doc_popart_popx_Executablex = R"doc()doc";

static const char *__doc_popart_popx_Executablex_2 = R"doc()doc";

static const char *__doc_popart_popx_Executablex_3 = R"doc()doc";

static const char *__doc_popart_popx_IrLowering = R"doc()doc";

static const char *__doc_popart_popx_IrLowering_2 = R"doc()doc";

static const char *__doc_popart_popx_exportExecutable = R"doc()doc";

static const char *__doc_popart_popx_exportStepIO = R"doc()doc";

static const char *__doc_popart_popx_exportStepIO_2 = R"doc()doc";

static const char *__doc_popart_popx_exportWeights = R"doc()doc";

static const char *__doc_popart_popx_exporterIsAvailable = R"doc()doc";

static const char *__doc_popart_popx_popType = R"doc()doc";

static const char *__doc_popart_popx_popType_2 = R"doc()doc";

static const char *__doc_popart_reservedAccl1Prefix = R"doc()doc";

static const char *__doc_popart_reservedAccl2Prefix = R"doc()doc";

static const char *__doc_popart_reservedAccl3Prefix = R"doc()doc";

static const char *__doc_popart_reservedAcclFinalOutPrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclPrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclToReducePrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclToUpdatePrefix = R"doc()doc";

static const char *__doc_popart_reservedAccumPrefix = R"doc()doc";

static const char *__doc_popart_reservedAccumulatorPrefixes = R"doc()doc";

static const char *__doc_popart_reservedAdamUpdaterPrefix = R"doc()doc";

static const char *__doc_popart_reservedAdaptiveUpdaterPrefix = R"doc()doc";

static const char *__doc_popart_reservedConcatInitPrefix = R"doc()doc";

static const char *__doc_popart_reservedConstValuePrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta1Prefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta2Prefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveAlphaPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveMomentumPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultDampeningScaleFactor1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultLearningRatePrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultMaxWeightNormPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledMomentum1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledWeightDecay1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayScaleFactor0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedGradientPrefix = R"doc()doc";

static const char *__doc_popart_reservedIndexPrefix = R"doc()doc";

static const char *__doc_popart_reservedInitPrefix = R"doc()doc";

static const char *__doc_popart_reservedLambR1SqPrefix = R"doc()doc";

static const char *__doc_popart_reservedLambR2SqPrefix = R"doc()doc";

static const char *__doc_popart_reservedLoopCondPrefix = R"doc()doc";

static const char *__doc_popart_reservedLoopIteratorPrefix = R"doc()doc";

static const char *__doc_popart_reservedLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedOptimizerPrefixes = R"doc()doc";

static const char *__doc_popart_reservedOptimizerStatePrefixes = R"doc()doc";

static const char *__doc_popart_reservedPrefixes = R"doc()doc";

static const char *__doc_popart_reservedRandomSeedPrefix = R"doc()doc";

static const char *__doc_popart_reservedRemoteArgPrefix = R"doc()doc";

static const char *__doc_popart_reservedRestoredPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta1Prefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta2Prefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveAlphaPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveMomentumPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificDampeningScaleFactor1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificLearningRatePrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificMaxWeightNormPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledMomentum1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledWeightDecay1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificWeightDecayPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificWeightDecayScaleFactor0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedStashedPrefix = R"doc()doc";

static const char *__doc_popart_reservedStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedUpdatedVarPrefix = R"doc()doc";

static const char *__doc_popart_stripAllReservedPrefixes = R"doc()doc";

static const char *__doc_popart_syncPatternFromString = R"doc()doc";

static const char *__doc_popart_syncPatternToString = R"doc()doc";

static const char *__doc_popart_toString = R"doc()doc";

static const char *__doc_popart_toString_2 = R"doc()doc";

static const char *__doc_poplar_Executable = R"doc()doc";

static const char *__doc_poplar_OptionFlags = R"doc()doc";

static const char *__doc_poplar_OptionFlags_2 = R"doc()doc";

static const char *__doc_poplar_Target = R"doc()doc";

static const char *__doc_std_hash = R"doc()doc";

static const char *__doc_std_hash_2 = R"doc()doc";

static const char *__doc_std_hash_3 = R"doc()doc";

static const char *__doc_std_hash_4 = R"doc()doc";

static const char *__doc_std_hash_5 = R"doc()doc";

static const char *__doc_std_hash_6 = R"doc()doc";

static const char *__doc_std_hash_7 = R"doc()doc";

static const char *__doc_std_hash_8 = R"doc()doc";

static const char *__doc_std_hash_9 = R"doc()doc";

static const char *__doc_std_hash_10 = R"doc()doc";

static const char *__doc_std_hash_operator_call = R"doc()doc";

static const char *__doc_std_hash_operator_call_2 = R"doc()doc";

static const char *__doc_std_hash_operator_call_3 = R"doc()doc";

static const char *__doc_std_hash_operator_call_4 = R"doc()doc";

static const char *__doc_std_hash_operator_call_5 = R"doc()doc";

static const char *__doc_std_hash_operator_call_6 = R"doc()doc";

static const char *__doc_std_hash_operator_call_7 = R"doc()doc";

static const char *__doc_std_hash_operator_call_8 = R"doc()doc";

static const char *__doc_std_hash_operator_call_9 = R"doc()doc";

static const char *__doc_std_hash_operator_call_10 = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
