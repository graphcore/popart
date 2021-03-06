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

static const char *__doc_gcl_CommGroup = R"doc()doc";

static const char *__doc_gcl_CommGroup_2 = R"doc()doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule =
    R"doc(Enum type that determines how the operations in the accumulate outer
fragment will be scheduled accross virtual graphs (only relevant to
pipelined modes).)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_2 =
    R"doc(Enum type that determines how the operations in the accumulate outer
fragment will be scheduled accross virtual graphs (only relevant to
pipelined modes).)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapCycleOptimized =
        R"doc(Try and parallelise ops with different virtual graph IDs as much as
possible.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapCycleOptimized_2 =
        R"doc(Try and parallelise ops with different virtual graph IDs as much as
possible.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapMemoryOptimized =
        R"doc(Try and parallelise ops with different virtual graph IDs but avoid
certain steps that are costly in terms of memory usage.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSchedule_OverlapMemoryOptimized_2 =
        R"doc(Try and parallelise ops with different virtual graph IDs but avoid
certain steps that are costly in terms of memory usage.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Scheduler =
    R"doc(Don't add additional constraints and let the scheduler work it out.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Scheduler_2 =
    R"doc(Don't add additional constraints and let the scheduler work it out.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Serial =
    R"doc(Add constraints that ensure ops are executed in virtual graph ID
order.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSchedule_Serial_2 =
    R"doc(Add constraints that ensure ops are executed in virtual graph ID
order.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings =
    R"doc(A structure containing accumulate outer fragment settings.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings_2 =
    R"doc(A structure containing accumulate outer fragment settings.)doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings =
        R"doc()doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings_2 =
        R"doc()doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings_3 =
        R"doc()doc";

static const char *
    __doc_popart_AccumulateOuterFragmentSettings_AccumulateOuterFragmentSettings_4 =
        R"doc()doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_excludedVirtualGraphs =
        R"doc(A setting to explicitly tell PopART to avoid to try and parallelise
the given virtual graph ids. This setting is experimental and may
change.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_excludedVirtualGraphs_2 =
        R"doc(A setting to explicitly tell PopART to avoid to try and parallelise
the given virtual graph ids. This setting is experimental and may
change.)doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_operator_assign = R"doc()doc";

static const char
    *__doc_popart_AccumulateOuterFragmentSettings_operator_assign_2 =
        R"doc()doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings_schedule =
    R"doc(Tell PopART how you would like to schedule the accumulate outer
fragment. This setting is experimental and may change.)doc";

static const char *__doc_popart_AccumulateOuterFragmentSettings_schedule_2 =
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

static const char *__doc_popart_Adam_2 =
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

static const char *__doc_popart_AdamMode_2 =
    R"doc(Enum type describing the mode of an Adam optimizer instance.)doc";

static const char *__doc_popart_AdamMode_AdaMax = R"doc(Adamax mode.)doc";

static const char *__doc_popart_AdamMode_AdaMax_2 = R"doc(Adamax mode.)doc";

static const char *__doc_popart_AdamMode_Adam =
    R"doc(Adam or AdamW mode, depending on weight decay setting (see [Kingma &
Ba, 2015](https://arxiv.org/abs/1412.6980) and [Loshchilov & Hutter,
2018](https://arxiv.org/pdf/1711.05101.pdf)).)doc";

static const char *__doc_popart_AdamMode_Adam_2 =
    R"doc(Adam or AdamW mode, depending on weight decay setting (see [Kingma &
Ba, 2015](https://arxiv.org/abs/1412.6980) and [Loshchilov & Hutter,
2018](https://arxiv.org/pdf/1711.05101.pdf)).)doc";

static const char *__doc_popart_AdamMode_AdamNoBias =
    R"doc(Like Adam but without bias correction.)doc";

static const char *__doc_popart_AdamMode_AdamNoBias_2 =
    R"doc(Like Adam but without bias correction.)doc";

static const char *__doc_popart_AdamMode_Lamb =
    R"doc(Lamb mode (see [You et al., 2020](https://arxiv.org/abs/1904.00962)).)doc";

static const char *__doc_popart_AdamMode_Lamb_2 =
    R"doc(Lamb mode (see [You et al., 2020](https://arxiv.org/abs/1904.00962)).)doc";

static const char *__doc_popart_AdamMode_LambNoBias =
    R"doc(Like Lamb but without bias correction.)doc";

static const char *__doc_popart_AdamMode_LambNoBias_2 =
    R"doc(Like Lamb but without bias correction.)doc";

static const char *__doc_popart_Adam_Adam =
    R"doc(Constructor.

Parameter ``defaultLearningRate``:
    The learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    The weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultBeta1``:
    The beta1 value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultBeta2``:
    The beta2 value value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultEps``:
    The epsilon value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``lossScaling``:
    The loss scaling value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``adamMode``:
    The AdamMode value to use.

Parameter ``weightDecayMode``:
    The WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``accumType``:
    Data type to use for gradient accumulation.

Parameter ``accl1Type``:
    Data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    Data type to use for tensor that stores second-order momentum
    optimizer state.)doc";

static const char *__doc_popart_Adam_Adam_2 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_3 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_4 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_5 =
    R"doc(Constructor.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"`, and the
    map's values pairs of floats and booleans representing
    OptimizerValue constructor arguments. The map does not have to
    specify each hyper parameter as default values will be used where
    parameters are missing.

Parameter ``adamMode``:
    The AdamMode value to use.

Parameter ``weightDecayMode``:
    The WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``accumType``:
    Data type to use for gradient accumulation.

Parameter ``accl1Type``:
    Data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    Data type to use for tensor that stores second-order momentum
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

static const char *__doc_popart_Adam_Adam_8 =
    R"doc(Constructor.

Parameter ``defaultLearningRate``:
    The learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    The weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultBeta1``:
    The beta1 value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultBeta2``:
    The beta2 value value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultEps``:
    The epsilon value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``lossScaling``:
    The loss scaling value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``adamMode``:
    The AdamMode value to use.

Parameter ``weightDecayMode``:
    The WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``accumType``:
    Data type to use for gradient accumulation.

Parameter ``accl1Type``:
    Data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    Data type to use for tensor that stores second-order momentum
    optimizer state.)doc";

static const char *__doc_popart_Adam_Adam_9 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_10 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_11 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_12 =
    R"doc(Constructor.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"`, and the
    map's values pairs of floats and booleans representing
    OptimizerValue constructor arguments. The map does not have to
    specify each hyper parameter as default values will be used where
    parameters are missing.

Parameter ``adamMode``:
    The AdamMode value to use.

Parameter ``weightDecayMode``:
    The WeightDecayMode value to use.

Parameter ``maxWeightNorm``:
    The maxWeightNorm value to use.

Parameter ``accumType``:
    Data type to use for gradient accumulation.

Parameter ``accl1Type``:
    Data type to use for tensor that stores first-order momentum
    optimizer state.

Parameter ``accl2Type``:
    Data type to use for tensor that stores second-order momentum
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

static const char *__doc_popart_Adam_Adam_13 = R"doc()doc";

static const char *__doc_popart_Adam_Adam_14 = R"doc()doc";

static const char *__doc_popart_Adam_accl1Type = R"doc()doc";

static const char *__doc_popart_Adam_accl1Type_2 = R"doc()doc";

static const char *__doc_popart_Adam_accl2Type = R"doc()doc";

static const char *__doc_popart_Adam_accl2Type_2 = R"doc()doc";

static const char *__doc_popart_Adam_accumType = R"doc()doc";

static const char *__doc_popart_Adam_accumType_2 = R"doc()doc";

static const char *__doc_popart_Adam_b1helper = R"doc()doc";

static const char *__doc_popart_Adam_b1helper_2 = R"doc()doc";

static const char *__doc_popart_Adam_b1s = R"doc()doc";

static const char *__doc_popart_Adam_b1s_2 = R"doc()doc";

static const char *__doc_popart_Adam_b2helper = R"doc()doc";

static const char *__doc_popart_Adam_b2helper_2 = R"doc()doc";

static const char *__doc_popart_Adam_b2s = R"doc()doc";

static const char *__doc_popart_Adam_b2s_2 = R"doc()doc";

static const char *__doc_popart_Adam_beta1s = R"doc()doc";

static const char *__doc_popart_Adam_beta1s_2 = R"doc()doc";

static const char *__doc_popart_Adam_beta2s = R"doc()doc";

static const char *__doc_popart_Adam_beta2s_2 = R"doc()doc";

static const char *__doc_popart_Adam_clone = R"doc()doc";

static const char *__doc_popart_Adam_clone_2 = R"doc()doc";

static const char *__doc_popart_Adam_createOp = R"doc()doc";

static const char *__doc_popart_Adam_createOp_2 = R"doc()doc";

static const char *__doc_popart_Adam_decayMode = R"doc()doc";

static const char *__doc_popart_Adam_decayMode_2 = R"doc()doc";

static const char *__doc_popart_Adam_epshelper = R"doc()doc";

static const char *__doc_popart_Adam_epshelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_epss = R"doc()doc";

static const char *__doc_popart_Adam_epss_2 = R"doc()doc";

static const char *__doc_popart_Adam_epsvs = R"doc()doc";

static const char *__doc_popart_Adam_epsvs_2 = R"doc()doc";

static const char *__doc_popart_Adam_fromDefaultMap = R"doc()doc";

static const char *__doc_popart_Adam_fromDefaultMap_2 = R"doc()doc";

static const char *__doc_popart_Adam_getComplete = R"doc()doc";

static const char *__doc_popart_Adam_getComplete_2 = R"doc()doc";

static const char *__doc_popart_Adam_getInputIds =
    R"doc(The names of the inputs for the VarUpdateOp for the Variable Tensor
"weight". In the returned vector, an empty string ("") is used as a
placeholder for constant inputs.)doc";

static const char *__doc_popart_Adam_getInputIds_2 =
    R"doc(The names of the inputs for the VarUpdateOp for the Variable Tensor
"weight". In the returned vector, an empty string ("") is used as a
placeholder for constant inputs.)doc";

static const char *__doc_popart_Adam_getOptimizerInputs =
    R"doc(The names and infos of the optimizer tensors.)doc";

static const char *__doc_popart_Adam_getOptimizerInputs_2 =
    R"doc(The names and infos of the optimizer tensors.)doc";

static const char *__doc_popart_Adam_getStoredValue =
    R"doc(Tensor "opt" has an id, based on which it matches a compound scalar
which this object can compute from the atomic scalars.)doc";

static const char *__doc_popart_Adam_getStoredValue_2 =
    R"doc(Tensor "opt" has an id, based on which it matches a compound scalar
which this object can compute from the atomic scalars.)doc";

static const char *__doc_popart_Adam_getUnsetBeta1 =
    R"doc(Default beta1 value.)doc";

static const char *__doc_popart_Adam_getUnsetBeta1_2 =
    R"doc(Default beta1 value.)doc";

static const char *__doc_popart_Adam_getUnsetBeta2 =
    R"doc(Default beta2 value.)doc";

static const char *__doc_popart_Adam_getUnsetBeta2_2 =
    R"doc(Default beta2 value.)doc";

static const char *__doc_popart_Adam_getUnsetEps =
    R"doc(Default epsilon value.)doc";

static const char *__doc_popart_Adam_getUnsetEps_2 =
    R"doc(Default epsilon value.)doc";

static const char *__doc_popart_Adam_getUnsetLearningRate =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_Adam_getUnsetLearningRate_2 =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_Adam_getUnsetLossScaling =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_Adam_getUnsetLossScaling_2 =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_Adam_getUnsetMaxWeightNorm =
    R"doc(Default maximum weight norm value.)doc";

static const char *__doc_popart_Adam_getUnsetMaxWeightNorm_2 =
    R"doc(Default maximum weight norm value.)doc";

static const char *__doc_popart_Adam_getUnsetWeightDecay =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_Adam_getUnsetWeightDecay_2 =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_Adam_gshelper = R"doc()doc";

static const char *__doc_popart_Adam_gshelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_hasSpecific = R"doc()doc";

static const char *__doc_popart_Adam_hasSpecific_2 = R"doc()doc";

static const char *__doc_popart_Adam_hash = R"doc()doc";

static const char *__doc_popart_Adam_hash_2 = R"doc()doc";

static const char *__doc_popart_Adam_insertSpecific =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``learningRate``:
    The learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    The weight decay value to use for this specific weight.

Parameter ``beta1``:
    The beta1 value to use for this specific weight.

Parameter ``beta2``:
    The beta2 value to use for this specific weight.

Parameter ``eps``:
    The epsilon value to use for this specific weight.)doc";

static const char *__doc_popart_Adam_insertSpecific_2 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"` and the map's
    values pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_Adam_insertSpecific_3 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``learningRate``:
    The learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    The weight decay value to use for this specific weight.

Parameter ``beta1``:
    The beta1 value to use for this specific weight.

Parameter ``beta2``:
    The beta2 value to use for this specific weight.

Parameter ``eps``:
    The epsilon value to use for this specific weight.)doc";

static const char *__doc_popart_Adam_insertSpecific_4 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultBeta1"`, `"defaultBeta2"`,
    `"defaultEps"`, `"lossScaling"` or `"maxWeightNorm"` and the map's
    values pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_Adam_learningRates = R"doc()doc";

static const char *__doc_popart_Adam_learningRates_2 = R"doc()doc";

static const char *__doc_popart_Adam_lrhelper = R"doc()doc";

static const char *__doc_popart_Adam_lrhelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_lrs = R"doc()doc";

static const char *__doc_popart_Adam_lrs_2 = R"doc()doc";

static const char *__doc_popart_Adam_lshelper = R"doc()doc";

static const char *__doc_popart_Adam_lshelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_maxWeightNorms = R"doc()doc";

static const char *__doc_popart_Adam_maxWeightNorms_2 = R"doc()doc";

static const char *__doc_popart_Adam_mode = R"doc()doc";

static const char *__doc_popart_Adam_mode_2 = R"doc()doc";

static const char *__doc_popart_Adam_mwnhelper = R"doc()doc";

static const char *__doc_popart_Adam_mwnhelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_mwns = R"doc()doc";

static const char *__doc_popart_Adam_mwns_2 = R"doc()doc";

static const char *__doc_popart_Adam_resetTensorData = R"doc()doc";

static const char *__doc_popart_Adam_resetTensorData_2 = R"doc()doc";

static const char *__doc_popart_Adam_runValueChecks = R"doc()doc";

static const char *__doc_popart_Adam_runValueChecks_2 = R"doc()doc";

static const char *__doc_popart_Adam_setStep = R"doc()doc";

static const char *__doc_popart_Adam_setStep_2 = R"doc()doc";

static const char *__doc_popart_Adam_setStep_3 = R"doc()doc";

static const char *__doc_popart_Adam_setStep_4 = R"doc()doc";

static const char *__doc_popart_Adam_setStep_5 = R"doc()doc";

static const char *__doc_popart_Adam_setStep_6 = R"doc()doc";

static const char *__doc_popart_Adam_setTensorData = R"doc()doc";

static const char *__doc_popart_Adam_setTensorData_2 = R"doc()doc";

static const char *__doc_popart_Adam_type = R"doc()doc";

static const char *__doc_popart_Adam_type_2 = R"doc()doc";

static const char *__doc_popart_Adam_type_s = R"doc()doc";

static const char *__doc_popart_Adam_type_s_2 = R"doc()doc";

static const char *__doc_popart_Adam_validReplacement = R"doc()doc";

static const char *__doc_popart_Adam_validReplacement_2 = R"doc()doc";

static const char *__doc_popart_Adam_wdhelper = R"doc()doc";

static const char *__doc_popart_Adam_wdhelper_2 = R"doc()doc";

static const char *__doc_popart_Adam_wds = R"doc()doc";

static const char *__doc_popart_Adam_wds_2 = R"doc()doc";

static const char *__doc_popart_Adam_weightDecays = R"doc()doc";

static const char *__doc_popart_Adam_weightDecays_2 = R"doc()doc";

static const char *__doc_popart_AddPatternName = R"doc()doc";

static const char *__doc_popart_AddPatternName_2 = R"doc()doc";

static const char *__doc_popart_AddPatternName_AddPatternName = R"doc()doc";

static const char *__doc_popart_AddPatternName_AddPatternName_2 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_2 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_AiGraphcoreOpset1 =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_AiGraphcoreOpset1_2 =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_abort =
    R"doc(Add abort operation to the model.

The operation can be conditional or unconditional.

Parameter ``args``:
    Optional input tensor to test condition)doc";

static const char *__doc_popart_AiGraphcoreOpset1_abort_2 =
    R"doc(Add abort operation to the model.

The operation can be conditional or unconditional.

Parameter ``args``:
    Optional input tensor to test condition)doc";

static const char *__doc_popart_AiGraphcoreOpset1_atan2 =
    R"doc(Add an ``atan2`` operation to the model.

Returns the element-wise angle theta as a tensor, -pi < theta <= pi,
such that for two input tensors x and y and given r != 0, x = r cos
theta, and y = r sin theta, element-wise.

In the case of x > 0, theta = arctan(y/x).

Parameter ``args``:
    Vector of input tensor ids: [y, x].

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor containing element wise theta
    values.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_atan2_2 =
    R"doc(Add an ``atan2`` operation to the model.

Returns the element-wise angle theta as a tensor, -pi < theta <= pi,
such that for two input tensors x and y and given r != 0, x = r cos
theta, and y = r sin theta, element-wise.

In the case of x > 0, theta = arctan(y/x).

Parameter ``args``:
    Vector of input tensor ids: [y, x].

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor containing element wise theta
    values.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseGenericOp =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseGenericOp_2 =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseand =
    R"doc(Add a bitwise AND operation to the model.

The operation computes the bitwise AND of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseand_2 =
    R"doc(Add a bitwise AND operation to the model.

The operation computes the bitwise AND of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisenot =
    R"doc(Add a bitwise NOT operation to the model.

The operation computes the bitwise NOT of a given integer tensor.

Parameter ``args``:
    Input tensor of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisenot_2 =
    R"doc(Add a bitwise NOT operation to the model.

The operation computes the bitwise NOT of a given integer tensor.

Parameter ``args``:
    Input tensor of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseor =
    R"doc(Add a bitwise OR operation to the model.

The operation computes the bitwise OR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwiseor_2 =
    R"doc(Add a bitwise OR operation to the model.

The operation computes the bitwise OR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisexnor =
    R"doc(Add a bitwise XNOR operation to the model.

The operation computes the bitwise XNOR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisexnor_2 =
    R"doc(Add a bitwise XNOR operation to the model.

The operation computes the bitwise XNOR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisexor =
    R"doc(Add a bitwise XOR operation to the model.

The operation computes the bitwise XOR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_bitwisexor_2 =
    R"doc(Add a bitwise XOR operation to the model.

The operation computes the bitwise XOR of given two integer tensors.

Parameter ``args``:
    Two broadcastable input tensors of type integer.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_call =
    R"doc(Add a call operation to the model

This is a Poplar extension, to expose manual code re-use to the
builder.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``callee``:
    The subgraph to call into.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A vector of tensors; the subgraph outputs.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_call_2 =
    R"doc(Add a call operation to the model

This is a Poplar extension, to expose manual code re-use to the
builder.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``callee``:
    The subgraph to call into.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A vector of tensors; the subgraph outputs.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_ctcloss =
    R"doc(Add a connectionist temporal classification (CTC) loss operation to
the model.

With T being maximum input length, N being batch size, C being number
of classes, S being a maximum target length, this op calculates the
CTC loss for a logarithmised probabilities tensor with shape [T, N,
C], a class target tensor with shape [N, S], an input lengths tensor
[N] and a target lengths tensor [N].

Note that C includes a blank class (default=0). The probabilities
tensor is padded as required. Target sequences are also padded and are
populated with values less than equal to C, not including the blank
class, up to their respective target lengths. Note that target lengths
cannot exceed input lengths.

Parameter ``args``:
    [log_probs,targets,input_lengths,target_lengths]

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``blank``:
    The integer representing the blank class.

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_ctcloss_2 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_ctcloss_3 =
    R"doc(Add a connectionist temporal classification (CTC) loss operation to
the model.

With T being maximum input length, N being batch size, C being number
of classes, S being a maximum target length, this op calculates the
CTC loss for a logarithmised probabilities tensor with shape [T, N,
C], a class target tensor with shape [N, S], an input lengths tensor
[N] and a target lengths tensor [N].

Note that C includes a blank class (default=0). The probabilities
tensor is padded as required. Target sequences are also padded and are
populated with values less than equal to C, not including the blank
class, up to their respective target lengths. Note that target lengths
cannot exceed input lengths.

Parameter ``args``:
    [log_probs,targets,input_lengths,target_lengths]

Parameter ``reduction``:
    Type of reduction to perform on the individual losses

Parameter ``blank``:
    The integer representing the blank class.

Parameter ``debugContext``:
    Optional debug context

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_ctcloss_4 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_depthtospace =
    R"doc(Add the ``DepthToSpace`` to the model. (This allows DepthToSpace_11 to
be targeted from earlier opsets.)

The purpose of Depth to Space, also known as pixel shuffling, is to
rearrange data from the depth (channels) dimension into the spacial
(width and height) dimensions. It is an efficient means of learning
upsampling alongside mixing convolution with bilinear interpolation
and using transpose convolution.

https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpac
e

Parameter ``args``:
    Vector containing single tensor input id.

Parameter ``blocksize``:
    Indicates the scale factor: if the input is [N, C, H, W] and the
    blocksize is B, the output will be [N, C/(B*B), H*B, W*B].

Parameter ``mode``:
    Specifies how the data is rearranged: * "DCR": depth-column-row
    order * "CRD": column-row-depth order

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A tensor which is a rearrangement of the input tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_depthtospace_2 =
    R"doc(Add the ``DepthToSpace`` to the model. (This allows DepthToSpace_11 to
be targeted from earlier opsets.)

The purpose of Depth to Space, also known as pixel shuffling, is to
rearrange data from the depth (channels) dimension into the spacial
(width and height) dimensions. It is an efficient means of learning
upsampling alongside mixing convolution with bilinear interpolation
and using transpose convolution.

https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpac
e

Parameter ``args``:
    Vector containing single tensor input id.

Parameter ``blocksize``:
    Indicates the scale factor: if the input is [N, C, H, W] and the
    blocksize is B, the output will be [N, C/(B*B), H*B, W*B].

Parameter ``mode``:
    Specifies how the data is rearranged: * "DCR": depth-column-row
    order * "CRD": column-row-depth order

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A tensor which is a rearrangement of the input tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_detach =
    R"doc(Add a detach operation to the model.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_detach_2 =
    R"doc(Add a detach operation to the model.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicadd =
    R"doc(Add a dynamic add operation to the model.

Creates a copy of ``tensor`` with ``slice`` added at ``offset``. For
example:

out = tensor, out[offset] += slice

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset, slice].

Parameter ``axes``:
    Axes along which to add.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicadd_2 =
    R"doc(Add a dynamic add operation to the model.

Creates a copy of ``tensor`` with ``slice`` added at ``offset``. For
example:

out = tensor, out[offset] += slice

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset, slice].

Parameter ``axes``:
    Axes along which to add.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicslice =
    R"doc(Add a dynamic slice operation to the model.

Creates a new slice tensor. For example:

slice = tensor[offset]

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset].

Parameter ``axes``:
    Axes along which to slice.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicslice_2 =
    R"doc(Add a dynamic slice operation to the model.

Creates a new slice tensor. For example:

slice = tensor[offset]

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset].

Parameter ``axes``:
    Axes along which to slice.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicupdate =
    R"doc(Add a dynamic update operation to the model.

Creates a copy of a ``tensor`` with a ``slice`` inserted at
``offset``. For example:

out = tensor, out[offset] = slice

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset, slice].

Parameter ``axes``:
    Axes along which to update.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamicupdate_2 =
    R"doc(Add a dynamic update operation to the model.

Creates a copy of a ``tensor`` with a ``slice`` inserted at
``offset``. For example:

out = tensor, out[offset] = slice

Parameter ``args``:
    Vector of input tensor ids: [tensor, offset, slice].

Parameter ``axes``:
    Axes along which to update.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamiczero =
    R"doc(Add a dynamic zero operation to the model.

Creates a copy of ``tensor`` with a slice at ``offset`` set to zero.
For example:

out = tensor, out[offset] = 0.0

Parameter ``args``:
    Vector of input tensor ids [tensor, offset].

Parameter ``axes``:
    Axes along which to erase.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_dynamiczero_2 =
    R"doc(Add a dynamic zero operation to the model.

Creates a copy of ``tensor`` with a slice at ``offset`` set to zero.
For example:

out = tensor, out[offset] = 0.0

Parameter ``args``:
    Vector of input tensor ids [tensor, offset].

Parameter ``axes``:
    Axes along which to erase.

Parameter ``sizes``:
    Size of the slice in each axis.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_expm1 =
    R"doc(Add ``expm1`` operation to the model. It computes exp(x) - 1.
Calculates the element-wise exponential of the input tensor and
subtracts one.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_expm1_2 =
    R"doc(Add ``expm1`` operation to the model. It computes exp(x) - 1.
Calculates the element-wise exponential of the input tensor and
subtracts one.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_fmod =
    R"doc(Add fmod operation to the model.

This is equivalent to C's fmod function. The result has the same sign
as the dividend.

Parameter ``args``:
    Input tensors.

Returns:
    Computes the element-wise remainder of division. The remainder has
    the same sign as the dividend.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_fmod_2 =
    R"doc(Add fmod operation to the model.

This is equivalent to C's fmod function. The result has the same sign
as the dividend.

Parameter ``args``:
    Input tensors.

Returns:
    Computes the element-wise remainder of division. The remainder has
    the same sign as the dividend.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_gelu =
    R"doc(Add a GELU operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_gelu_2 =
    R"doc(Add a GELU operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_getOpsetVersion_2 =
    R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_groupnormalization =
    R"doc(Add a group normalization operation to the model.

This is a Poplar extension.

The group will be created from a strided input.

Parameter ``args``:
    A vector of input tensors: [x, scale, bias].

Parameter ``num_groups``:
    The number of groups to separate the channels into.

Parameter ``epsilon``:
    The epsilon value to use to avoid division by zero.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A vector of tensors: [y, mean, var].)doc";

static const char *__doc_popart_AiGraphcoreOpset1_groupnormalization_2 =
    R"doc(Add a group normalization operation to the model.

This is a Poplar extension.

The group will be created from a strided input.

Parameter ``args``:
    A vector of input tensors: [x, scale, bias].

Parameter ``num_groups``:
    The number of groups to separate the channels into.

Parameter ``epsilon``:
    The epsilon value to use to avoid division by zero.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    A vector of tensors: [y, mean, var].)doc";

static const char *__doc_popart_AiGraphcoreOpset1_identityloss =
    R"doc(Add an identity loss operation to the model.

Calculates the loss using the identity operator.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_identityloss_2 =
    R"doc(Add an identity loss operation to the model.

Calculates the loss using the identity operator.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init =
    R"doc(Add an init operation to the model.

Parameter ``shape``:
    Shape of the tensor to initialise.

Parameter ``data_type``:
    Data type to initialise tensor with.

Parameter ``init_type``:
    Mode of tensor initialisations.

Parameter ``batch_axis``:
    Axis relative to batch size.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init_2 =
    R"doc(Add an init operation to the model.

Parameter ``shape``:
    Shape of the tensor to initialise.

Parameter ``data_type``:
    Data type to initialise tensor with.

Parameter ``init_type``:
    Mode of tensor initialisations.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init_3 =
    R"doc(Add an init operation to the model.

Parameter ``shape``:
    Shape of the tensor to initialise.

Parameter ``data_type``:
    Data type to initialise tensor with.

Parameter ``init_type``:
    Mode of tensor initialisations.

Parameter ``batch_axis``:
    Axis relative to batch size.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_init_4 =
    R"doc(Add an init operation to the model.

Parameter ``shape``:
    Shape of the tensor to initialise.

Parameter ``data_type``:
    Data type to initialise tensor with.

Parameter ``init_type``:
    Mode of tensor initialisations.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_l1loss =
    R"doc(Add an ``l1`` loss operation to the model.

Calculates the mean absolute error between each element in the input
with a zero target.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``lambda``:
    Scale factor of L1 loss.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_l1loss_2 =
    R"doc(Add an ``l1`` loss operation to the model.

Calculates the mean absolute error between each element in the input
with a zero target.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``lambda``:
    Scale factor of L1 loss.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_log1p =
    R"doc(Add ``log1p`` operation to the model. It computes log(x + 1). This
calculates the element-wise logarithm of the input tensor plus one.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_log1p_2 =
    R"doc(Add ``log1p`` operation to the model. It computes log(x + 1). This
calculates the element-wise logarithm of the input tensor plus one.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_lstm = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_lstm_2 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_multiconv =
    R"doc(Add a multi-convolution to the model.

Using this multi-convolution API ensures that the convolutions are
executed in parallel on the device.

Functionally, a multi-convolution is equivalent to a series of single
convolutions. Using this multi-convolution API is always equivalent to
calling the single-convolution API (conv) once for each argument.

For example, calling:

A0 = conv({X0, W0, B0}) A1 = conv({X1, W1})

Is functionally equivalent to calling:

{A0, A1} = multiconv({{X0, W0, B0}, {X1, Q1}).

It is possible that any two convolutions cannot be executed in
parallel due to topological constraints. For example, the following:

B = conv({A, W0}); C = B + A D = conv({C, W1});

Cannot be converted to:

{B, D} = multiconv({{A, W0}, {C, W1}}).

Note that it is not possible to create such a cycle by adding a multi-
convolution with this API.

Calls to multiconv() are mapped to
poplar::poplin::multiconv::convolution().

Parameter ``tensors``:
    List of [DataId, WeightId, BiasId (optional)] for each
    convolution.

Parameter ``dilations``:
    The dilations attributes for each convolution.

Parameter ``inDilations``:
    The input dilations attributes for each convolution.

Parameter ``pads``:
    The pads for each convolution.

Parameter ``outPads``:
    The output padding for each convolution.

Parameter ``strides``:
    The strides for each convolution.

Parameter ``availableMemoryProportions``:
    The available memory proportions per conv, each [0, 1).

Parameter ``partialsTypes``:
    The partials type per convolution.

Parameter ``planType``:
    Run convolutions in parallel or series.

Parameter ``perConvReservedTiles``:
    Tiles to reserve per convolution when planning.

Parameter ``cycleBackOff``:
    Cycle back-off proportion, [0, 1).

Parameter ``debugContext``:
    Optional debug context.

All input vectors must be either empty, or equal in length to the
number of convolutions. Note that groups for each convolution are
automatically inferred from the shapes of the data and weight inputs.

Returns:
    The TensorId of the output tensor from each convolution.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_multiconv_2 =
    R"doc(Add a multi-convolution to the model.

Using this multi-convolution API ensures that the convolutions are
executed in parallel on the device.

Functionally, a multi-convolution is equivalent to a series of single
convolutions. Using this multi-convolution API is always equivalent to
calling the single-convolution API (conv) once for each argument.

For example, calling:

A0 = conv({X0, W0, B0}) A1 = conv({X1, W1})

Is functionally equivalent to calling:

{A0, A1} = multiconv({{X0, W0, B0}, {X1, Q1}).

It is possible that any two convolutions cannot be executed in
parallel due to topological constraints. For example, the following:

B = conv({A, W0}); C = B + A D = conv({C, W1});

Cannot be converted to:

{B, D} = multiconv({{A, W0}, {C, W1}}).

Note that it is not possible to create such a cycle by adding a multi-
convolution with this API.

Calls to multiconv() are mapped to
poplar::poplin::multiconv::convolution().

Parameter ``tensors``:
    List of [DataId, WeightId, BiasId (optional)] for each
    convolution.

Parameter ``dilations``:
    The dilations attributes for each convolution.

Parameter ``inDilations``:
    The input dilations attributes for each convolution.

Parameter ``pads``:
    The pads for each convolution.

Parameter ``outPads``:
    The output padding for each convolution.

Parameter ``strides``:
    The strides for each convolution.

Parameter ``availableMemoryProportions``:
    The available memory proportions per conv, each [0, 1).

Parameter ``partialsTypes``:
    The partials type per convolution.

Parameter ``planType``:
    Run convolutions in parallel or series.

Parameter ``perConvReservedTiles``:
    Tiles to reserve per convolution when planning.

Parameter ``cycleBackOff``:
    Cycle back-off proportion, [0, 1).

Parameter ``debugContext``:
    Optional debug context.

All input vectors must be either empty, or equal in length to the
number of convolutions. Note that groups for each convolution are
automatically inferred from the shapes of the data and weight inputs.

Returns:
    The TensorId of the output tensor from each convolution.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nllloss =
    R"doc(Add a negative log-likelihood loss operation to the model.

Calculates the nll loss given a probability tensor over classes, and a
target tensor containing class labels.

Parameter ``args``:
    Vector of input tensor ids: probability and tensor.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``ignoreIndex``:
    Optional class index to ignore in loss calculation.

Parameter ``inputIsLogProbability``:
    Specifies if the input tensor contains log-probabilities or raw
    probabilities (false, default).

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nllloss_2 =
    R"doc(Add a negative log-likelihood loss operation to the model.

Calculates the nll loss given a probability tensor over classes, and a
target tensor containing class labels.

Parameter ``args``:
    Vector of input tensor ids: probability and tensor.

Parameter ``reduction``:
    Type of reduction to perform on the individual losses.

Parameter ``ignoreIndex``:
    Optional class index to ignore in loss calculation.

Parameter ``inputIsLogProbability``:
    Specifies if the input tensor contains log-probabilities or raw
    probabilities (false, default).

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nop =
    R"doc(Add a no-op operation to the model.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_nop_2 =
    R"doc(Add a no-op operation to the model.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_printtensor =
    R"doc(Add a print tensor operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of tensor ids to print.

Parameter ``print_gradient``:
    $Parameter ``debugContext``:

Optional debug context.

Parameter ``title``:
    $Returns:

The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_printtensor_2 =
    R"doc(Add a print tensor operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of tensor ids to print.

Parameter ``print_gradient``:
    $Parameter ``debugContext``:

Optional debug context.

Parameter ``title``:
    $Returns:

The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reducemedian = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_reducemedian_2 = R"doc()doc";

static const char *__doc_popart_AiGraphcoreOpset1_remainder =
    R"doc(Add remainder operation to the model.

This is equivalent to Python's modulo operator %. The result has the
same sign as the divisor.

Parameter ``args``:
    Input tensors.

Returns:
    Computes the element-wise remainder of division. The remainder has
    the same sign as the divisor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_remainder_2 =
    R"doc(Add remainder operation to the model.

This is equivalent to Python's modulo operator %. The result has the
same sign as the divisor.

Parameter ``args``:
    Input tensors.

Returns:
    Computes the element-wise remainder of division. The remainder has
    the same sign as the divisor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_replicatedallreduce =
    R"doc(Add a replicated all-reduce operation to the model.

This is a Poplar extension, to expose manual code re-use to the
builder.

Parameter ``args``:
    Vector of input tensor ids to reduce across.

Parameter ``commGroup``:
    GCL CommGroup parameter.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_replicatedallreduce_2 =
    R"doc(Add a replicated all-reduce operation to the model.

This is a Poplar extension, to expose manual code re-use to the
builder.

Parameter ``args``:
    Vector of input tensor ids to reduce across.

Parameter ``commGroup``:
    GCL CommGroup parameter.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reshape =
    R"doc(Add reshape operation to the model. Reshape the input tensor. This
reshape takes the shape to reshape into as an attribute instead of a
tensor input as the ONNX reshape op.

Parameter ``arg``:
    Vector with single input tensor id.

Parameter ``shape``:
    The shape of the output Tensor. The output Tensor must contain the
    same number of elements as the input Tensor.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reshape_2 =
    R"doc(Add reshape operation to the model. Reshape the input tensor. This
reshape takes the shape to reshape into as an attribute instead of a
tensor input as the ONNX reshape op.

Parameter ``arg``:
    Vector with single input tensor id.

Parameter ``shape``:
    The shape of the output Tensor. The output Tensor must contain the
    same number of elements as the input Tensor.

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reverse =
    R"doc(Add a reverse operator to the model.

Reverse, or 'flip', the tensor along the specified dimensions

Parameter ``args``:
    Input tensors.

Parameter ``dimensions``:
    Dimensions along which to reverse the tensor. If this is empty
    then this is equivalent to the identity operator

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_reverse_2 =
    R"doc(Add a reverse operator to the model.

Reverse, or 'flip', the tensor along the specified dimensions

Parameter ``args``:
    Input tensors.

Parameter ``dimensions``:
    Dimensions along which to reverse the tensor. If this is empty
    then this is equivalent to the identity operator

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_round =
    R"doc(Add a ``Round`` operation to the model. (This allows ``Round_11`` to
be targeted from earlier opsets.)

https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The normalized output tensor ids.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_round_2 =
    R"doc(Add a ``Round`` operation to the model. (This allows ``Round_11`` to
be targeted from earlier opsets.)

https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The normalized output tensor ids.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scale =
    R"doc(Add a scale operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``scale``:
    The scale to apply.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scale_2 =
    R"doc(Add a scale operation to the model.

This is a Poplar extension.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``scale``:
    The scale to apply.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scaledadd =
    R"doc(Add a scaled add operation to the model.

X = scale0 * T0 + scale1 * T1

Parameter ``args``:
    Vector of input tensor ids: [T0, T1, scale0, scale1].

Parameter ``scale0``:
    The scale to apply (if no ``scale0`` tensor is supplied).

Parameter ``scale1``:
    The scale to apply (if no ``scale1`` tensor is supplied).

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scaledadd_2 =
    R"doc(Add a scaled add operation to the model.

X = scale0 * T0 + scale1 * T1

Parameter ``args``:
    Vector of input tensor ids: [T0, T1, scale0, scale1].

Parameter ``scale0``:
    The scale to apply (if no ``scale0`` tensor is supplied).

Parameter ``scale1``:
    The scale to apply (if no ``scale1`` tensor is supplied).

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scatterreduce =
    R"doc(Add a scatterreduce operation to the model

Reduces all the values from the src tensor at the indices specified
along the given axis.

for i in range(axis_size): output[i] = reduce(src[index == i])

Parameter ``args``:
    list of [src, index] tensors

Parameter ``axis_size``:
    Size in the reduced axis

Parameter ``axis``:
    Axis to reduce along (default = -1)

Parameter ``reduction``:
    The type of reduction to apply (default = "sum")

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_scatterreduce_2 =
    R"doc(Add a scatterreduce operation to the model

Reduces all the values from the src tensor at the indices specified
along the given axis.

for i in range(axis_size): output[i] = reduce(src[index == i])

Parameter ``args``:
    list of [src, index] tensors

Parameter ``axis_size``:
    Size in the reduced axis

Parameter ``axis``:
    Axis to reduce along (default = -1)

Parameter ``reduction``:
    The type of reduction to apply (default = "sum")

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_sequenceslice =
    R"doc(Slice a 2D tensor based on offsets specified by a tensor.

The outermost dimension is sliced; tOut[tOutOffset:tOutOffset+tN][...]
= tIn[tInOffset:tInOffset+tN][...] for each entry in
tN/tInOffset/tOutOffset; entries after the first tN==0 may be ignored.
Unreferenced elements of tOut are zeroed if zeroUnused is set. The
same output element should not be written by multiple inputs.

tIn and tOut must have rank greater than or equal to 2. The outer
dimension is sliced; the product of the inner dimensions must match.
tInOffset, tOutOffset and tN must be 1d and the same size. \param
[source, destination, N, sourceOffset, destinationOffset]

Parameter ``zeroUnused``:
    Whether to zero unreferenced tOut elements.

Parameter ``debugContext``:
    Optional debug context.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_sequenceslice_2 =
    R"doc(Slice a 2D tensor based on offsets specified by a tensor.

The outermost dimension is sliced; tOut[tOutOffset:tOutOffset+tN][...]
= tIn[tInOffset:tInOffset+tN][...] for each entry in
tN/tInOffset/tOutOffset; entries after the first tN==0 may be ignored.
Unreferenced elements of tOut are zeroed if zeroUnused is set. The
same output element should not be written by multiple inputs.

tIn and tOut must have rank greater than or equal to 2. The outer
dimension is sliced; the product of the inner dimensions must match.
tInOffset, tOutOffset and tN must be 1d and the same size. \param
[source, destination, N, sourceOffset, destinationOffset]

Parameter ``zeroUnused``:
    Whether to zero unreferenced tOut elements.

Parameter ``debugContext``:
    Optional debug context.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_shapeddropout =
    R"doc(Add a shaped dropout operation to the model.

Applies a shaped dropout to the input tensor. This operator requires a
shape parameter that is used to define the shape of the dropout mask
so that strongly correlated features in the input tensor can be
preserved. The provided shape must be broadcastable to the input
tensor. Note that this operation targets the poprand library function
of the same name.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``shape``:
    Shape of dropout mask. Must be broadcastable to the input.

Parameter ``ratio``:
    Probability of dropping an input feature (default = 0.5).

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_shapeddropout_2 =
    R"doc(Add a shaped dropout operation to the model.

Applies a shaped dropout to the input tensor. This operator requires a
shape parameter that is used to define the shape of the dropout mask
so that strongly correlated features in the input tensor can be
preserved. The provided shape must be broadcastable to the input
tensor. Note that this operation targets the poprand library function
of the same name.

Parameter ``args``:
    Vector of input tensor ids.

Parameter ``shape``:
    Shape of dropout mask. Must be broadcastable to the input.

Parameter ``ratio``:
    Probability of dropping an input feature (default = 0.5).

Parameter ``name``:
    Optional identifier for operation.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_subsample =
    R"doc(Add a sub-sample operation to the model.

This is a Poplar extension.

If multiple tensors are provided that strides will applied to them
all.

Parameter ``args``:
    Vector of tensor ids to sub-sample.

Parameter ``strides``:
    The strides to use.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiGraphcoreOpset1_subsample_2 =
    R"doc(Add a sub-sample operation to the model.

This is a Poplar extension.

If multiple tensors are provided that strides will applied to them
all.

Parameter ``args``:
    Vector of tensor ids to sub-sample.

Parameter ``strides``:
    The strides to use.

Parameter ``debugContext``:
    Optional debug context.

Returns:
    The name of the result tensor.)doc";

static const char *__doc_popart_AiOnnxMlOpset1 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_2 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_AiOnnxMlOpset1 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_AiOnnxMlOpset1_2 = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_AiOnnxMlOpset1_getOpsetVersion_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType =
    R"doc(A class that captures an #AnchorReturnTypeId value and, when this
value is ``AnchorReturnTypeId::EVERYN``, the associated *N* value. The
constructor takes `std::string` values and converts them as
appropriate.)doc";

static const char *__doc_popart_AnchorReturnType_2 =
    R"doc(A class that captures an #AnchorReturnTypeId value and, when this
value is ``AnchorReturnTypeId::EVERYN``, the associated *N* value. The
constructor takes `std::string` values and converts them as
appropriate.)doc";

static const char *__doc_popart_AnchorReturnTypeId =
    R"doc(An anchor tensor is a tensor that the user wants returned after a call
to Session::run(). Each call to Session::run() results in
`batchesPerStep x accumulationFactor x replicationFactor` of such
tensors being computed. We refer to the samples associated with each
such computation as a micro batch. The dimensions are user-specified
by the following parameters:

* `batchesPerStep` is the value in DataFlow. * `accumulationFactor` is
the value defined by SessionOptions::accumulationFactor. *
`replicationFactor` is the value defined by
SessionOptions::replicatedGraphCount.

This enum type describes the strategy with which the micro batch
values for anchor tensors (or summaries thereof) are written or to the
IStepIO instance passed to Session::run.

See also: AnchorReturnType.

**NOTE**: Anchors are essentially what TensorFlow calls "fetches".)doc";

static const char *__doc_popart_AnchorReturnTypeId_2 =
    R"doc(An anchor tensor is a tensor that the user wants returned after a call
to Session::run(). Each call to Session::run() results in
`batchesPerStep x accumulationFactor x replicationFactor` of such
tensors being computed. We refer to the samples associated with each
such computation as a micro batch. The dimensions are user-specified
by the following parameters:

* `batchesPerStep` is the value in DataFlow. * `accumulationFactor` is
the value defined by SessionOptions::accumulationFactor. *
`replicationFactor` is the value defined by
SessionOptions::replicatedGraphCount.

This enum type describes the strategy with which the micro batch
values for anchor tensors (or summaries thereof) are written or to the
IStepIO instance passed to Session::run.

See also: AnchorReturnType.

**NOTE**: Anchors are essentially what TensorFlow calls "fetches".)doc";

static const char *__doc_popart_AnchorReturnTypeId_All =
    R"doc(Return the tensor value for *all* micro batches for each replica.

The buffer shape required for this anchor in IStepIO is
[`batchesPerStep`, `accumulationFactor`, `replicationFactor`,
`<anchorTensorShape>`] (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_All_2 =
    R"doc(Return the tensor value for *all* micro batches for each replica.

The buffer shape required for this anchor in IStepIO is
[`batchesPerStep`, `accumulationFactor`, `replicationFactor`,
`<anchorTensorShape>`] (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_EveryN =
    R"doc(Return the tensor value for every *N*\ th global batch for each
replica and for all accumulation steps in that global batch. Note that
the value of *N* is captured by AnchorReturnType.

The buffer shape required for this anchor in IStepIO is
[`batchesPerStep // N`, `accumulationFactor`, `replicationFactor`,
`<anchorTensorShape>`] (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_EveryN_2 =
    R"doc(Return the tensor value for every *N*\ th global batch for each
replica and for all accumulation steps in that global batch. Note that
the value of *N* is captured by AnchorReturnType.

The buffer shape required for this anchor in IStepIO is
[`batchesPerStep // N`, `accumulationFactor`, `replicationFactor`,
`<anchorTensorShape>`] (with dimensions of size 1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Final =
    R"doc(Only return the tensor value for the last micro batch of the
Session::run call for each replica.

The buffer shape required for this anchor in IStepIO is
[`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size
1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Final_2 =
    R"doc(Only return the tensor value for the last micro batch of the
Session::run call for each replica.

The buffer shape required for this anchor in IStepIO is
[`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size
1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Sum =
    R"doc(Return one tensor value for each replica, doing a sum reduction over
the `batchesPerStep` and `accumulationFactor` dimensions.

The buffer shape required for this anchor in IStepIO is
[`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size
1 removed).)doc";

static const char *__doc_popart_AnchorReturnTypeId_Sum_2 =
    R"doc(Return one tensor value for each replica, doing a sum reduction over
the `batchesPerStep` and `accumulationFactor` dimensions.

The buffer shape required for this anchor in IStepIO is
[`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size
1 removed).)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType =
    R"doc(Constructor.

Parameter ``artString``:
    - the string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * "final" =
    ``AnchorReturnTypeId::FINAL`` * "all" =
    ``AnchorReturnTypeId::ALL`` * "sum" = ``AnchorReturnTypeId::SUM``

**NOTE**: Attempting to construct an AnchorReturnType for
``AnchorReturnTypeId::EVERYN`` using this constructor will result in
an error. Use the constructor that also specifies the return period.)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType_2 =
    R"doc(Constructor.

Parameter ``artString``:
    The string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * "final" =
    ``AnchorReturnTypeId::FINAL`` * "everyn" =
    ``AnchorReturnTypeId::EVERYN`` * "all" =
    ``AnchorReturnTypeId::ALL`` * "sum" = ``AnchorReturnTypeId::SUM``

Parameter ``returnPeriod``:
    The value of *N* in the case of ``AnchorReturnTypeId::EVERYN``.)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType_3 =
    R"doc(Constructor.

Parameter ``artString``:
    - the string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * "final" =
    ``AnchorReturnTypeId::FINAL`` * "all" =
    ``AnchorReturnTypeId::ALL`` * "sum" = ``AnchorReturnTypeId::SUM``

**NOTE**: Attempting to construct an AnchorReturnType for
``AnchorReturnTypeId::EVERYN`` using this constructor will result in
an error. Use the constructor that also specifies the return period.)doc";

static const char *__doc_popart_AnchorReturnType_AnchorReturnType_4 =
    R"doc(Constructor.

Parameter ``artString``:
    The string to convert to an #AnchorReturnTypeId value. The
    following values are acceptable (case insensitive): * "final" =
    ``AnchorReturnTypeId::FINAL`` * "everyn" =
    ``AnchorReturnTypeId::EVERYN`` * "all" =
    ``AnchorReturnTypeId::ALL`` * "sum" = ``AnchorReturnTypeId::SUM``

Parameter ``returnPeriod``:
    The value of *N* in the case of ``AnchorReturnTypeId::EVERYN``.)doc";

static const char *__doc_popart_AnchorReturnType_artId = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_artId_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_artStr = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_artStr_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_getIdFromStr = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_getIdFromStr_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_hash = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_hash_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_id =
    R"doc(Return the associated #AnchorReturnTypeId, not currently part of
public API.)doc";

static const char *__doc_popart_AnchorReturnType_id_2 =
    R"doc(Return the associated #AnchorReturnTypeId, not currently part of
public API.)doc";

static const char *__doc_popart_AnchorReturnType_returnPeriod = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_returnPeriod_2 = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_rp =
    R"doc(Return the associated return period (*N*) if the #AnchorReturnTypeId
is ``AnchorReturnTypeId::EVERYN``, not currently part of public API.)doc";

static const char *__doc_popart_AnchorReturnType_rp_2 =
    R"doc(Return the associated return period (*N*) if the #AnchorReturnTypeId
is ``AnchorReturnTypeId::EVERYN``, not currently part of public API.)doc";

static const char *__doc_popart_AnchorReturnType_str = R"doc()doc";

static const char *__doc_popart_AnchorReturnType_str_2 = R"doc()doc";

static const char *__doc_popart_BatchSerializationBatchSchedule =
    R"doc(Enum type that describes how to change the batch serialisation
subgraph schedule before outlining. **NOTE:** This setting is
experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_2 =
    R"doc(Enum type that describes how to change the batch serialisation
subgraph schedule before outlining. **NOTE:** This setting is
experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Isomorphic =
    R"doc(Encourage all ops within batch subgraphs to be scheduled identically
and for each subgraph to be scheduled in sequence (good for
outlineability).)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Isomorphic_2 =
    R"doc(Encourage all ops within batch subgraphs to be scheduled identically
and for each subgraph to be scheduled in sequence (good for
outlineability).)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_N =
    R"doc(The number of ``BatchSerializationBatchSchedule`` values.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_N_2 =
    R"doc(The number of ``BatchSerializationBatchSchedule`` values.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_OverlapOnCompute =
    R"doc(Attempt to put the RemoteLoad for batch N+1 right before the compute
phase of batch N.)doc";

static const char
    *__doc_popart_BatchSerializationBatchSchedule_OverlapOnCompute_2 =
        R"doc(Attempt to put the RemoteLoad for batch N+1 right before the compute
phase of batch N.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_OverlapOnIo =
    R"doc(Attempt to put the RemoteLoad for batch N+1 right after the compute
phase of batch N.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_OverlapOnIo_2 =
    R"doc(Attempt to put the RemoteLoad for batch N+1 right after the compute
phase of batch N.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Scheduler =
    R"doc(Don't encourage any particular scheduling for ops within batch
subgraphs (leave it to the scheduler) but tell the scheduler to
schedule subgraphs in sequence.)doc";

static const char *__doc_popart_BatchSerializationBatchSchedule_Scheduler_2 =
    R"doc(Don't encourage any particular scheduling for ops within batch
subgraphs (leave it to the scheduler) but tell the scheduler to
schedule subgraphs in sequence.)doc";

static const char *__doc_popart_BatchSerializationMethod =
    R"doc(Enum type that describes how to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationMethod_2 =
    R"doc(Enum type that describes how to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationMethod_Loop =
    R"doc(Loop over the batch dimension)doc";

static const char *__doc_popart_BatchSerializationMethod_Loop_2 =
    R"doc(Loop over the batch dimension)doc";

static const char *__doc_popart_BatchSerializationMethod_N =
    R"doc(The number of ``BatchSerializationMethod`` values.)doc";

static const char *__doc_popart_BatchSerializationMethod_N_2 =
    R"doc(The number of ``BatchSerializationMethod`` values.)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollDynamic =
    R"doc(Unroll the batch with dynamic slicing)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollDynamic_2 =
    R"doc(Unroll the batch with dynamic slicing)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollStatic =
    R"doc(Unroll the batch with static slicing)doc";

static const char *__doc_popart_BatchSerializationMethod_UnrollStatic_2 =
    R"doc(Unroll the batch with static slicing)doc";

static const char *__doc_popart_BatchSerializationSettings =
    R"doc(A structure containing batch serialization settings.)doc";

static const char *__doc_popart_BatchSerializationSettings_2 =
    R"doc(A structure containing batch serialization settings.)doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings =
        R"doc()doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings_2 =
        R"doc()doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings_3 =
        R"doc()doc";

static const char
    *__doc_popart_BatchSerializationSettings_BatchSerializationSettings_4 =
        R"doc()doc";

static const char *__doc_popart_BatchSerializationSettings_batchSchedule =
    R"doc(Experimental value that changes how operations are scheduled.)doc";

static const char *__doc_popart_BatchSerializationSettings_batchSchedule_2 =
    R"doc(Experimental value that changes how operations are scheduled.)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnExecutionPhaseChange =
        R"doc(Break batch serialization chains when the execution phase changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnExecutionPhaseChange_2 =
        R"doc(Break batch serialization chains when the execution phase changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnPipelineStageChange =
        R"doc(Break batch serialization chains when the pipeline stage changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnPipelineStageChange_2 =
        R"doc(Break batch serialization chains when the pipeline stage changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnVirtualGraphChange =
        R"doc(Break batch serialization chains when the virtual graph changes (by
concatenating the compute batches to the local batch).)doc";

static const char
    *__doc_popart_BatchSerializationSettings_concatOnVirtualGraphChange_2 =
        R"doc(Break batch serialization chains when the virtual graph changes (by
concatenating the compute batches to the local batch).)doc";

static const char *__doc_popart_BatchSerializationSettings_factor =
    R"doc(The number of compute batches to split operations into.)doc";

static const char *__doc_popart_BatchSerializationSettings_factor_2 =
    R"doc(The number of compute batches to split operations into.)doc";

static const char *__doc_popart_BatchSerializationSettings_method =
    R"doc(Experimental value to control how batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationSettings_method_2 =
    R"doc(Experimental value to control how batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationSettings_operator_assign =
    R"doc()doc";

static const char *__doc_popart_BatchSerializationSettings_operator_assign_2 =
    R"doc()doc";

static const char *__doc_popart_BatchSerializationSettings_transformContext =
    R"doc(Experimental value to control when batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationSettings_transformContext_2 =
    R"doc(Experimental value to control when batch serialization is applied.)doc";

static const char *__doc_popart_BatchSerializationTransformContext =
    R"doc(Enum type that describes when to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationTransformContext_2 =
    R"doc(Enum type that describes when to apply the batch serialization.
**NOTE:** This setting is experimental and may change.)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Bwd =
    R"doc(Apply after growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Bwd_2 =
    R"doc(Apply after growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Fwd =
    R"doc(Apply before growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_Fwd_2 =
    R"doc(Apply before growing the backward pass)doc";

static const char *__doc_popart_BatchSerializationTransformContext_N =
    R"doc(The number of ``BatchSerializationTransformContext`` values.)doc";

static const char *__doc_popart_BatchSerializationTransformContext_N_2 =
    R"doc(The number of ``BatchSerializationTransformContext`` values.)doc";

static const char *__doc_popart_Builder =
    R"doc(\class Builder A builder interface for creating ONNX graphs.

ONNX defines a specification for describing graphs and serialising
them as protobuf files. This class provides a builder interface for
creating such a graph.

Note, in ONNX, all Ops belong to an "Opset". The Builder itself does
not have methods for creating Ops in the ONNX graph, but instead has
accessors to Opsets, like AiGraphcoreOpset1, which contain the methods
for creating Ops in the graph.)doc";

static const char *__doc_popart_Builder_2 =
    R"doc(\class Builder A builder interface for creating ONNX graphs.

ONNX defines a specification for describing graphs and serialising
them as protobuf files. This class provides a builder interface for
creating such a graph.

Note, in ONNX, all Ops belong to an "Opset". The Builder itself does
not have methods for creating Ops in the ONNX graph, but instead has
accessors to Opsets, like AiGraphcoreOpset1, which contain the methods
for creating Ops in the graph.)doc";

static const char *__doc_popart_Builder_3 =
    R"doc(\class Builder A builder interface for creating ONNX graphs.

ONNX defines a specification for describing graphs and serialising
them as protobuf files. This class provides a builder interface for
creating such a graph.

Note, in ONNX, all Ops belong to an "Opset". The Builder itself does
not have methods for creating Ops in the ONNX graph, but instead has
accessors to Opsets, like AiGraphcoreOpset1, which contain the methods
for creating Ops in the graph.)doc";

static const char *__doc_popart_Builder_4 =
    R"doc(\class Builder A builder interface for creating ONNX graphs.

ONNX defines a specification for describing graphs and serialising
them as protobuf files. This class provides a builder interface for
creating such a graph.

Note, in ONNX, all Ops belong to an "Opset". The Builder itself does
not have methods for creating Ops in the ONNX graph, but instead has
accessors to Opsets, like AiGraphcoreOpset1, which contain the methods
for creating Ops in the graph.)doc";

static const char *__doc_popart_BuilderImpl = R"doc()doc";

static const char *__doc_popart_BuilderImpl_2 = R"doc()doc";

static const char *__doc_popart_Builder_Builder = R"doc()doc";

static const char *__doc_popart_Builder_Builder_2 = R"doc()doc";

static const char *__doc_popart_Builder_addInitializedInputTensor =
    R"doc(Add a new pre-initialized input tensor to the model.

Parameter ``initData``:
    The initial data of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInitializedInputTensor_2 =
    R"doc(Add a new pre-initialized input tensor to the model.

Parameter ``initData``:
    The initial data of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInputTensor =
    R"doc(Add a new input tensor to the model.

Parameter ``tensorInfo``:
    The shape and type of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInputTensor_2 =
    R"doc(Add a new input tensor to the model.

Parameter ``dataType``:
    The type of the input tensor.

Parameter ``shape``:
    The shape of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInputTensor_3 =
    R"doc(Add a new input tensor to the model.

Parameter ``tensorInfo``:
    The shape and type of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInputTensor_4 =
    R"doc(Add a new input tensor to the model.

Parameter ``dataType``:
    The type of the input tensor.

Parameter ``shape``:
    The shape of the input tensor.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addInputTensorFromParentGraph =
    R"doc(Add a new named input tensor to the model.

Parameter ``tensorId``:
    The identifier string of the input tensor. This identifier must
    already exist in the parent GraphProto's name scope and must
    appear topologically before this sub-graph.)doc";

static const char *__doc_popart_Builder_addInputTensorFromParentGraph_2 =
    R"doc(Add a new named input tensor to the model.

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
    An ``int64_t`` value of the attribute to add.

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
    A ``std::vector``<int64_t> value of the attribute to add.

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
    A ``float`` value of the attribute to add.

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
    The ``std::vector``<float> value of the attribute to add.

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
    A ``std::string`` value of the attribute to add.

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
    A ``std::vector``<std::string> value of the attribute to add.

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
    A bool value of the attribute to add.

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
    A constant tensor initializer.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_10 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    An ``int64_t`` value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_11 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A ``std::vector``<int64_t> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_12 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A ``float`` value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_13 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    The ``std::vector``<float> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_14 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A ``std::string`` value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_15 = R"doc()doc";

static const char *__doc_popart_Builder_addNodeAttribute_16 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A ``std::vector``<std::string> value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_17 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A bool value of the attribute to add.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addNodeAttribute_18 =
    R"doc(Add an attribute to the ONNX node which is uniquely identified by the
outputs. This functions will throw an exception if it can't find the
unique node or the attribute already exists.

Parameter ``attributeName``:
    The name of the attribute to add.

Parameter ``attributeValue``:
    A constant tensor initializer.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_addOutputTensor =
    R"doc(Adds one of the outputs from a node in the graph into the list of
output tensors.)doc";

static const char *__doc_popart_Builder_addOutputTensor_2 =
    R"doc(Adds one of the outputs from a node in the graph into the list of
output tensors.)doc";

static const char *__doc_popart_Builder_addUntypedInputTensor =
    R"doc(Add a new input tensor without a type or shape to the model.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_addUntypedInputTensor_2 =
    R"doc(Add a new input tensor without a type or shape to the model.

Parameter ``debugContext``:
    Optional debug information.

Returns:
    The unique name of the input tensor.)doc";

static const char *__doc_popart_Builder_aiGraphcoreOpset1 =
    R"doc(Return the builder interface for ai.graphcore opset 1.)doc";

static const char *__doc_popart_Builder_aiGraphcoreOpset1_2 =
    R"doc(Return the builder interface for ai.graphcore opset 1.)doc";

static const char *__doc_popart_Builder_aiOnnxMlOpset1 =
    R"doc(Return the builder interface for ai.onnx.ml opset 1.)doc";

static const char *__doc_popart_Builder_aiOnnxMlOpset1_2 =
    R"doc(Return the builder interface for ai.onnx.ml opset 1.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset10 =
    R"doc(Return the builder interface for ai.onnx opset 10.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset10_2 =
    R"doc(Return the builder interface for ai.onnx opset 10.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset11 =
    R"doc(Return the builder interface for ai.onnx opset 11.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset11_2 =
    R"doc(Return the builder interface for ai.onnx opset 11.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset6 =
    R"doc(Return the builder interface for ai.onnx opset 6.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset6_2 =
    R"doc(Return the builder interface for ai.onnx opset 6.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset7 =
    R"doc(Return the builder interface for ai.onnx opset 7.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset7_2 =
    R"doc(Return the builder interface for ai.onnx opset 7.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset8 =
    R"doc(Return the builder interface for ai.onnx opset 7.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset8_2 =
    R"doc(Return the builder interface for ai.onnx opset 7.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset9 =
    R"doc(Return the builder interface for ai.onnx opset 9.)doc";

static const char *__doc_popart_Builder_aiOnnxOpset9_2 =
    R"doc(Return the builder interface for ai.onnx opset 9.)doc";

static const char *__doc_popart_Builder_checkpointOutput =
    R"doc(Add checkpoint operations to the model.

This is the same as an identity but is recomputeType Checkpoint by
default. Use this to checkpoint a subset of an operation's output
tensors.

Parameter ``nodeOutputNames``:
    Tensors to checkpoint.

Returns:
    The checkpointed tensors.)doc";

static const char *__doc_popart_Builder_checkpointOutput_2 =
    R"doc(Add checkpoint operations to the model.

This is the same as an identity but is recomputeType Checkpoint by
default. Use this to checkpoint a subset of an operation's output
tensors.

Parameter ``nodeOutputNames``:
    Tensors to checkpoint.

Returns:
    The checkpointed tensors.)doc";

static const char *__doc_popart_Builder_children = R"doc()doc";

static const char *__doc_popart_Builder_children_2 = R"doc()doc";

static const char *__doc_popart_Builder_clearAttribute =
    R"doc(Unset an attribute that will be set on all subsequent operations.)doc";

static const char *__doc_popart_Builder_clearAttribute_2 =
    R"doc(Unset an attribute that will be set on all subsequent operations.)doc";

static const char *__doc_popart_Builder_configure = R"doc()doc";

static const char *__doc_popart_Builder_configure_2 = R"doc()doc";

static const char *__doc_popart_Builder_configure_3 = R"doc()doc";

static const char *__doc_popart_Builder_configure_4 = R"doc()doc";

static const char *__doc_popart_Builder_create =
    R"doc(Create a builder for an ONNX model.)doc";

static const char *__doc_popart_Builder_create_2 =
    R"doc(Create a builder for an ONNX model.)doc";

static const char *__doc_popart_Builder_createFromOnnxModel =
    R"doc(Create a builder which loads a serialized ONNX ModelProto into the
builder and validates it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_createFromOnnxModel_2 =
    R"doc(Create a builder which loads a serialized ONNX ModelProto into the
builder and validates it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_createSubgraphBuilder =
    R"doc(Return a Builder for a graph which is nested inside this Builder's
graph.)doc";

static const char *__doc_popart_Builder_createSubgraphBuilder_2 =
    R"doc(Return a Builder for a graph which is nested inside this Builder's
graph.)doc";

static const char *__doc_popart_Builder_customOp = R"doc()doc";

static const char *__doc_popart_Builder_customOp_2 = R"doc()doc";

static const char *__doc_popart_Builder_customOp_3 = R"doc()doc";

static const char *__doc_popart_Builder_customOp_4 = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns_2 = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns_3 = R"doc()doc";

static const char *__doc_popart_Builder_excludePatterns_4 = R"doc()doc";

static const char *__doc_popart_Builder_executionPhase =
    R"doc(Set the execution phase that computes the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_Builder_executionPhase_2 = R"doc()doc";

static const char *__doc_popart_Builder_executionPhase_3 =
    R"doc(Set the execution phase that computes the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_Builder_executionPhase_4 = R"doc()doc";

static const char *__doc_popart_Builder_getAllNodeAttributeNames =
    R"doc(Get all the attribute names from the ONNX node. This functions will
throw an exception if it can't find the unique node.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getAllNodeAttributeNames_2 =
    R"doc(Get all the attribute names from the ONNX node. This functions will
throw an exception if it can't find the unique node.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getAttribute =
    R"doc(Get an attribute that has been set for all subsequent operations.)doc";

static const char *__doc_popart_Builder_getAttribute_2 =
    R"doc(Get the current attribute value.)doc";

static const char *__doc_popart_Builder_getAttribute_3 =
    R"doc(Get an attribute that has been set for all subsequent operations.)doc";

static const char *__doc_popart_Builder_getAttribute_4 =
    R"doc(Get the current attribute value.)doc";

static const char *__doc_popart_Builder_getBoolNodeAttribute = R"doc()doc";

static const char *__doc_popart_Builder_getBoolNodeAttribute_2 = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase =
    R"doc(A convenience function for getting the execution phase attribute.)doc";

static const char *__doc_popart_Builder_getExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase_3 = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase_4 =
    R"doc(A convenience function for getting the execution phase attribute.)doc";

static const char *__doc_popart_Builder_getExecutionPhase_5 = R"doc()doc";

static const char *__doc_popart_Builder_getExecutionPhase_6 = R"doc()doc";

static const char *__doc_popart_Builder_getFloatNodeAttribute =
    R"doc(Get the ``float`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the ``float``
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getFloatNodeAttribute_2 =
    R"doc(Get the ``float`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the ``float``
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getFloatVectorNodeAttribute =
    R"doc(Get the ``std::vector``<float> value of the attribute for the ONNX
node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getFloatVectorNodeAttribute_2 =
    R"doc(Get the ``std::vector``<float> value of the attribute for the ONNX
node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getInputTensorIds =
    R"doc(Return a list of ONNX graph input tensor ids.

Returns:
    A vector of input tensor names.)doc";

static const char *__doc_popart_Builder_getInputTensorIds_2 =
    R"doc(Return a list of ONNX graph input tensor ids.

Returns:
    A vector of input tensor names.)doc";

static const char *__doc_popart_Builder_getInt64NodeAttribute =
    R"doc(Get the ``int64_t`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the ``int64_t``
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getInt64NodeAttribute_2 =
    R"doc(Get the ``int64_t`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the ``int64_t``
type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getInt64VectorNodeAttribute =
    R"doc(Get the ``std::vector``<int64_t> value of the attribute for the ONNX
node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist or it has not been set to
the ``std::vector``<int64_t> type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getInt64VectorNodeAttribute_2 =
    R"doc(Get the ``std::vector``<int64_t> value of the attribute for the ONNX
node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist or it has not been set to
the ``std::vector``<int64_t> type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getModelProto =
    R"doc(Retrieve the ONNX serialized ModelProto.

Returns:
    A serialized ONNX ModelProto.)doc";

static const char *__doc_popart_Builder_getModelProto_2 =
    R"doc(Retrieve the ONNX serialized ModelProto.

Returns:
    A serialized ONNX ModelProto.)doc";

static const char *__doc_popart_Builder_getNameScope =
    R"doc(Get the current namescope stack using the default delimiter.

Parameter ``name``:
    Optional string to concatenate to the end of the stack

Returns:
    A string of the concatenated namescope stack.)doc";

static const char *__doc_popart_Builder_getNameScope_2 =
    R"doc(Get the current namescope stack using the default delimiter.

Parameter ``name``:
    Optional string to concatenate to the end of the stack

Returns:
    A string of the concatenated namescope stack.)doc";

static const char *__doc_popart_Builder_getOutputTensorIds =
    R"doc(Return a list of ONNX graph output tensor ids.

Returns:
    A vector of output tensor names.)doc";

static const char *__doc_popart_Builder_getOutputTensorIds_2 =
    R"doc(Return a list of ONNX graph output tensor ids.

Returns:
    A vector of output tensor names.)doc";

static const char *__doc_popart_Builder_getParent =
    R"doc(Returns the parent graph of this graph or null if there is no parent.)doc";

static const char *__doc_popart_Builder_getParent_2 =
    R"doc(Returns the parent graph of this graph or null if there is no parent.)doc";

static const char *__doc_popart_Builder_getPartialsType =
    R"doc(Get the partials type for the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.)doc";

static const char *__doc_popart_Builder_getPartialsType_2 =
    R"doc(Get the partials type for the given node.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.)doc";

static const char *__doc_popart_Builder_getPipelineStage =
    R"doc(A convenience function for getting the pipeline stage attribute.)doc";

static const char *__doc_popart_Builder_getPipelineStage_2 =
    R"doc(A convenience function for getting the pipeline stage attribute.)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass =
    R"doc(Return whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass_2 =
    R"doc(Return whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass_3 =
    R"doc(Return whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getRecomputeOutputInBackwardPass_4 =
    R"doc(Return whether the given node will have its output recomputed in the
backward pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getStringNodeAttribute =
    R"doc(Get the ``std::string`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the
``std::string`` type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getStringNodeAttribute_2 =
    R"doc(Get the ``std::string`` value of the attribute for the ONNX node. This
functions will throw an exception if it can't find the unique node or
the attribute does not exist or it has not been set to the
``std::string`` type.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getStringVectorNodeAttribute =
    R"doc(Get the ``std::vector``<std::string> value of the attribute for the
ONNX node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getStringVectorNodeAttribute_2 =
    R"doc(Get the ``std::vector``<std::string> value of the attribute for the
ONNX node. This functions will throw an exception if it can't find the
unique node or the attribute does not exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.

Returns:
    Value of the attribute.)doc";

static const char *__doc_popart_Builder_getTensorDataType =
    R"doc(Return a tensor type from either the input, output, or value_info
lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A tensor type.)doc";

static const char *__doc_popart_Builder_getTensorDataType_2 =
    R"doc(Return a tensor type from either the input, output, or value_info
lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A tensor type.)doc";

static const char *__doc_popart_Builder_getTensorDtypeString =
    R"doc(Return an ONNX graph tensor type as a lower case string, from either
the input, output, or value_info lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A lower case string of tensor type.)doc";

static const char *__doc_popart_Builder_getTensorDtypeString_2 =
    R"doc(Return an ONNX graph tensor type as a lower case string, from either
the input, output, or value_info lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A lower case string of tensor type.)doc";

static const char *__doc_popart_Builder_getTensorShape =
    R"doc(Return an ONNX graph tensor shape, from either the input, output, or
value_info lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A vector of tensor dimensions.)doc";

static const char *__doc_popart_Builder_getTensorShape_2 =
    R"doc(Return an ONNX graph tensor shape, from either the input, output, or
value_info lists in the GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A vector of tensor dimensions.)doc";

static const char *__doc_popart_Builder_getTrainableTensorIds =
    R"doc(Return a list of ONNX graph initialized tensor ids.

These tensors are stored in the `initialized` section of the ONNX
GraphProto structure..

Returns:
    A vector of tensor names.)doc";

static const char *__doc_popart_Builder_getTrainableTensorIds_2 =
    R"doc(Return a list of ONNX graph initialized tensor ids.

These tensors are stored in the `initialized` section of the ONNX
GraphProto structure..

Returns:
    A vector of tensor names.)doc";

static const char *__doc_popart_Builder_getValueTensorIds =
    R"doc(Return a list of ONNX graph value tensor ids.

These tensors are stored in the `value_info` section of the ONNX
GraphProto structure.

Returns:
    A vector of output tensor names.)doc";

static const char *__doc_popart_Builder_getValueTensorIds_2 =
    R"doc(Return a list of ONNX graph value tensor ids.

These tensors are stored in the `value_info` section of the ONNX
GraphProto structure.

Returns:
    A vector of output tensor names.)doc";

static const char *__doc_popart_Builder_getVirtualGraph =
    R"doc(A convenience function for getting the virtual graph attribute.)doc";

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

static const char *__doc_popart_Builder_getVirtualGraph_4 =
    R"doc(A convenience function for getting the virtual graph attribute.)doc";

static const char *__doc_popart_Builder_getVirtualGraph_5 =
    R"doc(Get the index of the virtual graph that computes this node. This
applies in a multi IPU system.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_getVirtualGraph_6 =
    R"doc(Get the index of the virtual graph that computes this node. This
applies in a multi IPU system.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_hasAttribute = R"doc()doc";

static const char *__doc_popart_Builder_hasAttribute_2 =
    R"doc(Check if an attribute is set.)doc";

static const char *__doc_popart_Builder_hasAttribute_3 = R"doc()doc";

static const char *__doc_popart_Builder_hasAttribute_4 =
    R"doc(Check if an attribute is set.)doc";

static const char *__doc_popart_Builder_hasParent =
    R"doc(Returns true if this builder represents a subgraph.)doc";

static const char *__doc_popart_Builder_hasParent_2 =
    R"doc(Returns true if this builder represents a subgraph.)doc";

static const char *__doc_popart_Builder_impl = R"doc()doc";

static const char *__doc_popart_Builder_impl_2 = R"doc()doc";

static const char *__doc_popart_Builder_isInitializer =
    R"doc(Returns true if the ONNX tensor is in the initializer list of the
GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A boolean.)doc";

static const char *__doc_popart_Builder_isInitializer_2 =
    R"doc(Returns true if the ONNX tensor is in the initializer list of the
GraphProto.

Parameter ``id``:
    Tensor id.

Returns:
    A boolean.)doc";

static const char *__doc_popart_Builder_loadModelProto =
    R"doc(Load a serialized ONNX ModelProto into the builder and validate it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_loadModelProto_2 =
    R"doc(Load a serialized ONNX ModelProto into the builder and validate it.

Parameter ``modelProtoOrFilename``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_nChildren = R"doc()doc";

static const char *__doc_popart_Builder_nChildren_2 = R"doc()doc";

static const char *__doc_popart_Builder_nodeHasAttribute =
    R"doc(Check whether the ONNX node has an attribute set. This functions will
throw an exception if it can't find the unique node.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_nodeHasAttribute_2 =
    R"doc(Check whether the ONNX node has an attribute set. This functions will
throw an exception if it can't find the unique node.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_outputTensorLocation = R"doc()doc";

static const char *__doc_popart_Builder_outputTensorLocation_2 = R"doc()doc";

static const char *__doc_popart_Builder_parent = R"doc()doc";

static const char *__doc_popart_Builder_parent_2 = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage_3 = R"doc()doc";

static const char *__doc_popart_Builder_pipelineStage_4 = R"doc()doc";

static const char *__doc_popart_Builder_popNameScope =
    R"doc(Remove the last entry in the name scope stack.)doc";

static const char *__doc_popart_Builder_popNameScope_2 =
    R"doc(Remove the last entry in the name scope stack.)doc";

static const char *__doc_popart_Builder_pushNameScope =
    R"doc(Push a name onto the name scope stack.

The names of tensors and nodes added to the ONNX graph will be
prefixed with a concatenation of the names in the name stack.)doc";

static const char *__doc_popart_Builder_pushNameScope_2 =
    R"doc(Push a name onto the name scope stack.

The names of tensors and nodes added to the ONNX graph will be
prefixed with a concatenation of the names in the name stack.)doc";

static const char *__doc_popart_Builder_recomputeOutput = R"doc()doc";

static const char *__doc_popart_Builder_recomputeOutput_2 = R"doc()doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    If the recompute is enabled/disabled.)doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass_2 =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node.

Parameter ``value``:
    If the recompute is enabled/disabled.)doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass_3 =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    If the recompute is enabled/disabled.)doc";

static const char *__doc_popart_Builder_recomputeOutputInBackwardPass_4 =
    R"doc(Enable/disable recomputation of the output of the node in the backward
pass.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node.

Parameter ``value``:
    If the recompute is enabled/disabled.)doc";

static const char *__doc_popart_Builder_removeNodeAttribute =
    R"doc(Remove an attribute from the ONNX node. This functions will throw an
exception if it can't find the unique node or the attribute does not
exist.

Parameter ``attributeName``:
    The name of the attribute to find.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node used to find the node
    in the ONNX model.)doc";

static const char *__doc_popart_Builder_removeNodeAttribute_2 =
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

static const char *__doc_popart_Builder_reshape_const_2 =
    R"doc(This is a helper function that will add a constant and a reshape using
the provided domain.)doc";

static const char *__doc_popart_Builder_saveInitializersExternally =
    R"doc(Save tensor data externally.

The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To avoid this, for large models ONNX tensor data can be saved
separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data. This can be
    an absolute or relative path. If a relative path, when the ONNX
    model is saved, external tensor data will be written to a path
    relative to your current working directory.)doc";

static const char *__doc_popart_Builder_saveInitializersExternally_2 =
    R"doc(Save tensor data externally.

The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To avoid this, for large models ONNX tensor data can be saved
separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data. This can be
    an absolute or relative path. If a relative path, when the ONNX
    model is saved, external tensor data will be written to a path
    relative to your current working directory.)doc";

static const char *__doc_popart_Builder_saveModelProto =
    R"doc(Save the builder's ONNX ModelProto into the builder and validate it.

Parameter ``fn``:
    The name of a file containing an ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_saveModelProto_2 =
    R"doc(Save the builder's ONNX ModelProto into the builder and validate it.

Parameter ``fn``:
    The name of a file containing an ONNX model protobuf.)doc";

static const char *__doc_popart_Builder_setAttribute =
    R"doc(Set an attribute that will be set on all subsequent operations.)doc";

static const char *__doc_popart_Builder_setAttribute_2 =
    R"doc(Set an attribute that will be set on all subsequent operations.)doc";

static const char *__doc_popart_Builder_setAvailableMemoryProportion =
    R"doc(Set the available memory for the given node. Used on the convolution
op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``availableMemoryProportion``:
    The available memory proportion 0 < x <= 1.)doc";

static const char *__doc_popart_Builder_setAvailableMemoryProportion_2 =
    R"doc(Set the available memory for the given node. Used on the convolution
op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``availableMemoryProportion``:
    The available memory proportion 0 < x <= 1.)doc";

static const char *__doc_popart_Builder_setGraphName =
    R"doc(Specifies a graph name.

Parameter ``name``:
    String to name the graph.)doc";

static const char *__doc_popart_Builder_setGraphName_2 =
    R"doc(Specifies a graph name.

Parameter ``name``:
    String to name the graph.)doc";

static const char *__doc_popart_Builder_setInplacePreferences = R"doc()doc";

static const char *__doc_popart_Builder_setInplacePreferences_2 = R"doc()doc";

static const char *__doc_popart_Builder_setParent =
    R"doc(Sets the parent graph of this builder.

Parameter ``parent``:
    the builder to become a parent.)doc";

static const char *__doc_popart_Builder_setParent_2 =
    R"doc(Sets the parent graph of this builder.

Parameter ``parent``:
    the builder to become a parent.)doc";

static const char *__doc_popart_Builder_setPartialsType =
    R"doc(Set the partials type for the given node. Used on the convolution op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``partialsType``:
    The type for the partials. Can be either FLOAT or HALF.)doc";

static const char *__doc_popart_Builder_setPartialsType_2 =
    R"doc(Set the partials type for the given node. Used on the convolution op.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``partialsType``:
    The type for the partials. Can be either FLOAT or HALF.)doc";

static const char *__doc_popart_Builder_setSerializeMatMul =
    R"doc(Set the settings for matmuls that should be serialized. This option
will split a matmul into separate smaller matmuls that will be
executed in series. This will also serialize the grad operations if
training.

Parameter ``nodeOutputNames``:
    Name of the output matmul tensors of the ONNX node.

Parameter ``mode``:
    Which dimension of the mat mul to serialize on.

Parameter ``factor``:
    The number of serialised matmuls, must be a factor of the
    dimensions to serialise on.)doc";

static const char *__doc_popart_Builder_setSerializeMatMul_2 =
    R"doc(Set the settings for matmuls that should be serialized. This option
will split a matmul into separate smaller matmuls that will be
executed in series. This will also serialize the grad operations if
training.

Parameter ``nodeOutputNames``:
    Name of the output matmul tensors of the ONNX node.

Parameter ``mode``:
    Which dimension of the mat mul to serialize on.

Parameter ``factor``:
    The number of serialised matmuls, must be a factor of the
    dimensions to serialise on.)doc";

static const char *__doc_popart_Builder_virtualGraph =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_Builder_virtualGraph_2 =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_Builder_virtualGraph_3 =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputName``:
    Name of the output tensor of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_Builder_virtualGraph_4 =
    R"doc(Set the virtual graph that computes the given node. Applies when
creating a graph for a multi-IPU configuration.

Parameter ``nodeOutputNames``:
    Names of the output tensors of the ONNX node.

Parameter ``value``:
    The index of the virtual graph that computes this node.)doc";

static const char *__doc_popart_ClipNormSettings =
    R"doc(A data structure used to represent a maximum value constaint on one or
more weights.)doc";

static const char *__doc_popart_ClipNormSettings_2 =
    R"doc(A data structure used to represent a maximum value constaint on one or
more weights.)doc";

static const char *__doc_popart_ClipNormSettings_ClipNormSettings =
    R"doc(Constructor.

Parameter ``weightIds_``:
    The weight tensor IDs that this constraint applies to.

Parameter ``maxNorm_``:
    The maximum permissible value.)doc";

static const char *__doc_popart_ClipNormSettings_ClipNormSettings_2 =
    R"doc(Constructor.

Parameter ``weightIds_``:
    The weight tensor IDs that this constraint applies to.

Parameter ``maxNorm_``:
    The maximum permissible value.)doc";

static const char *__doc_popart_ClipNormSettings_maxNorm = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_maxNorm_2 = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_eq = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_eq_2 = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_ne = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_operator_ne_2 = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_weightIds = R"doc()doc";

static const char *__doc_popart_ClipNormSettings_weightIds_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Add = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Add_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Local = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Local_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_LogicalAnd = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_LogicalAnd_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_LogicalOr = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_LogicalOr_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Max = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Max_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Min = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Min_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Mul = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_Mul_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_N = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_N_2 = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_SquareAdd = R"doc()doc";

static const char *__doc_popart_CollectiveOperator_SquareAdd_2 = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_2 = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_CollectivesBaseOp =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_CollectivesBaseOp_2 =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_appendOutlineAttributes =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_appendOutlineAttributes_2 =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getCollectiveLinkedIndex =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getCollectiveLinkedIndex_2 =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getGCLCommGroup = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getGCLCommGroup_2 =
    R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getInIndex = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getInIndex_2 = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_group = R"doc()doc";

static const char *__doc_popart_CollectivesBaseOp_group_2 = R"doc()doc";

static const char *__doc_popart_CommGroup =
    R"doc(Struct to specify sub-groups of replicas.

Examples of derived sub-groups: - IPU-link domain sub-rack:

```
type == Consecutive && replicaGroupSize == 64/replica-size/N
```

where N is power of two and replicaGroupSize > 1. - Complete IPU-link
domain / full rack:

```
type == Consecutive && replicaGroupSize == 64/replica-size
```

- Using GW-links only:

```
type == Orthogonal && replicaGroupSize == 64/replica-size
```)doc";

static const char *__doc_popart_CommGroup_2 =
    R"doc(Struct to specify sub-groups of replicas.

Examples of derived sub-groups: - IPU-link domain sub-rack:

```
type == Consecutive && replicaGroupSize == 64/replica-size/N
```

where N is power of two and replicaGroupSize > 1. - Complete IPU-link
domain / full rack:

```
type == Consecutive && replicaGroupSize == 64/replica-size
```

- Using GW-links only:

```
type == Orthogonal && replicaGroupSize == 64/replica-size
```)doc";

static const char *__doc_popart_CommGroupType =
    R"doc(PopART equivalent of GCL CommGroupType. Each of these enumeration
constants have a corresponding GCL CommGroupType value.)doc";

static const char *__doc_popart_CommGroupType_2 =
    R"doc(PopART equivalent of GCL CommGroupType. Each of these enumeration
constants have a corresponding GCL CommGroupType value.)doc";

static const char *__doc_popart_CommGroupType_All =
    R"doc(All replicas viewed as one group, replica group size is ignored. */)doc";

static const char *__doc_popart_CommGroupType_All_2 =
    R"doc(All replicas viewed as one group, replica group size is ignored. */)doc";

static const char *__doc_popart_CommGroupType_Consecutive =
    R"doc(Groups are consecutive in replica. If there are N replicas denoted
{0....N-1} and group size is k then the groups are: {0, 1, ... k-1},
{k, ... 2k-1} ... {N-k-1, ... N-1})doc";

static const char *__doc_popart_CommGroupType_Consecutive_2 =
    R"doc(Groups are consecutive in replica. If there are N replicas denoted
{0....N-1} and group size is k then the groups are: {0, 1, ... k-1},
{k, ... 2k-1} ... {N-k-1, ... N-1})doc";

static const char *__doc_popart_CommGroupType_N =
    R"doc(Groups are sliced orthogonal to the replica ordering. If there are N
replicas denoted {0....N-1} and group size is k then the groups are:
{0, k, 2k, ...}, {1, k+1, 2k+1, ...} ... {k-1, 2k-1, ..., N-1})doc";

static const char *__doc_popart_CommGroupType_N_2 =
    R"doc(Groups are sliced orthogonal to the replica ordering. If there are N
replicas denoted {0....N-1} and group size is k then the groups are:
{0, k, 2k, ...}, {1, k+1, 2k+1, ...} ... {k-1, 2k-1, ..., N-1})doc";

static const char *__doc_popart_CommGroupType_Orthogonal =
    R"doc(Groups are sliced orthogonal to the replica ordering. If there are N
replicas denoted {0....N-1} and group size is k then the groups are:
{0, k, 2k, ...}, {1, k+1, 2k+1, ...} ... {k-1, 2k-1, ..., N-1})doc";

static const char *__doc_popart_CommGroupType_Orthogonal_2 =
    R"doc(Groups are sliced orthogonal to the replica ordering. If there are N
replicas denoted {0....N-1} and group size is k then the groups are:
{0, k, 2k, ...}, {1, k+1, 2k+1, ...} ... {k-1, 2k-1, ..., N-1})doc";

static const char *__doc_popart_CommGroup_CommGroup = R"doc()doc";

static const char *__doc_popart_CommGroup_CommGroup_2 =
    R"doc(Construct CommGroup

Parameter ``groupType``:
    replica group type

Parameter ``groupSize``:
    replica group size)doc";

static const char *__doc_popart_CommGroup_CommGroup_3 = R"doc()doc";

static const char *__doc_popart_CommGroup_CommGroup_4 =
    R"doc(Construct CommGroup

Parameter ``groupType``:
    replica group type

Parameter ``groupSize``:
    replica group size)doc";

static const char *__doc_popart_CommGroup_replicaGroupSize =
    R"doc(Replica group size */)doc";

static const char *__doc_popart_CommGroup_replicaGroupSize_2 =
    R"doc(Replica group size */)doc";

static const char *__doc_popart_CommGroup_type =
    R"doc(Replica group type */)doc";

static const char *__doc_popart_CommGroup_type_2 =
    R"doc(Replica group type */)doc";

static const char *__doc_popart_ConstSGD =
    R"doc(Stochastic Gradient Descent (SGD) optimizer with constant learning
rate, weight decay, loss scaling and clip norm settings (and default
values for momentum, dampening or velocity scaling).

**NOTE**: See SGD for detailed meaning for these parameters.

**NOTE**: This class exists for backwards compatibility with the
Python API and may be removed at some point in the future.)doc";

static const char *__doc_popart_ConstSGD_2 =
    R"doc(Stochastic Gradient Descent (SGD) optimizer with constant learning
rate, weight decay, loss scaling and clip norm settings (and default
values for momentum, dampening or velocity scaling).

**NOTE**: See SGD for detailed meaning for these parameters.

**NOTE**: This class exists for backwards compatibility with the
Python API and may be removed at some point in the future.)doc";

static const char *__doc_popart_ConstSGD_ConstSGD =
    R"doc(Constructor.

Parameter ``learningRate``:
    A constant learning rate.

Parameter ``weightDecay``:
    A constant weight decay value.

Parameter ``lossScaling``:
    A constant loss scaling value.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_ConstSGD_ConstSGD_2 =
    R"doc(Constructor.

Parameter ``learningRate``:
    A constant learning rate.

Parameter ``weightDecay``:
    A constant weight decay value.

Parameter ``lossScaling``:
    A constant loss scaling value.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_Consumers = R"doc()doc";

static const char *__doc_popart_Consumers_Consumers = R"doc()doc";

static const char *__doc_popart_Consumers_append = R"doc()doc";

static const char *__doc_popart_Consumers_consumers_m = R"doc()doc";

static const char *__doc_popart_Consumers_decrement = R"doc()doc";

static const char *__doc_popart_Consumers_extend = R"doc()doc";

static const char *__doc_popart_Consumers_findHighestPipelineStage =
    R"doc()doc";

static const char *__doc_popart_Consumers_findLowestPipelineStage = R"doc()doc";

static const char *__doc_popart_Consumers_getMap = R"doc()doc";

static const char *__doc_popart_Consumers_getOps = R"doc()doc";

static const char *__doc_popart_Consumers_getPipelineStages = R"doc()doc";

static const char *__doc_popart_Consumers_getTotal = R"doc()doc";

static const char *__doc_popart_Consumers_increment = R"doc()doc";

static const char *__doc_popart_Consumers_n = R"doc()doc";

static const char *__doc_popart_Consumers_tensorConsumed = R"doc()doc";

static const char *__doc_popart_DataFlow =
    R"doc(This class specifies parameters for host-device data streams. The
parameters are used to control the amount input data processed each
step (that is: each Session::run call) determines how data is returned
to the user.

See also: AnchorReturnType, #AnchorReturnTypeId.)doc";

static const char *__doc_popart_DataFlow_2 =
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
    The number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorMap``:
    A mapping from output tensor TensorId to AnchorReturnType
    indicating the strategy with which to write the anchor tensor
    values to the IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_4 =
    R"doc(Constructor DataFlow instance with anchor tensors.

Parameter ``batchesPerStep``:
    The number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorTensorIds``:
    The tensor ID of anchor tensors.

Parameter ``anchorReturnType``:
    The strategy with which to write anchor tensor values to the
    IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_5 = R"doc()doc";

static const char *__doc_popart_DataFlow_DataFlow_6 =
    R"doc(Default constructor, sets `batchesPerStep` to 0 and does not have any
anchors.)doc";

static const char *__doc_popart_DataFlow_DataFlow_7 =
    R"doc(Construct DataFlow instance without anchor tensors.

Parameter ``batchesPerStep``:
    - the number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.)doc";

static const char *__doc_popart_DataFlow_DataFlow_8 =
    R"doc(Constructor DataFlow instance with anchor tensors.

Parameter ``batchesPerStep``:
    The number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorMap``:
    A mapping from output tensor TensorId to AnchorReturnType
    indicating the strategy with which to write the anchor tensor
    values to the IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_9 =
    R"doc(Constructor DataFlow instance with anchor tensors.

Parameter ``batchesPerStep``:
    The number of global batches to run the inference or training
    session for per call to Session::run before returning control to
    the caller.

Parameter ``anchorTensorIds``:
    The tensor ID of anchor tensors.

Parameter ``anchorReturnType``:
    The strategy with which to write anchor tensor values to the
    IStepIO object provided to Session::run.)doc";

static const char *__doc_popart_DataFlow_DataFlow_10 = R"doc()doc";

static const char *__doc_popart_DataFlow_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_anchors_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_art = R"doc()doc";

static const char *__doc_popart_DataFlow_art_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep_3 = R"doc()doc";

static const char *__doc_popart_DataFlow_batchesPerStep_4 = R"doc()doc";

static const char *__doc_popart_DataFlow_getAnchorMap = R"doc()doc";

static const char *__doc_popart_DataFlow_getAnchorMap_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_hash = R"doc()doc";

static const char *__doc_popart_DataFlow_hash_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_isAnchored = R"doc()doc";

static const char *__doc_popart_DataFlow_isAnchored_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_isBatchCountingRequired = R"doc()doc";

static const char *__doc_popart_DataFlow_isBatchCountingRequired_2 =
    R"doc()doc";

static const char *__doc_popart_DataFlow_isValidAnchorReturnPeriod =
    R"doc()doc";

static const char *__doc_popart_DataFlow_isValidAnchorReturnPeriod_2 =
    R"doc()doc";

static const char *__doc_popart_DataFlow_m_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_m_anchors_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_nAnchors = R"doc()doc";

static const char *__doc_popart_DataFlow_nAnchors_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_numOutFetchesPerRepl = R"doc()doc";

static const char *__doc_popart_DataFlow_numOutFetchesPerRepl_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_operator_assign = R"doc()doc";

static const char *__doc_popart_DataFlow_operator_assign_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_rps = R"doc()doc";

static const char *__doc_popart_DataFlow_rps_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_s_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_s_anchors_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_v_anchors = R"doc()doc";

static const char *__doc_popart_DataFlow_v_anchors_2 = R"doc()doc";

static const char *__doc_popart_DataFlow_v_rps = R"doc()doc";

static const char *__doc_popart_DataFlow_v_rps_2 = R"doc()doc";

static const char *__doc_popart_DataType =
    R"doc(There is a one-to-one correspondence between ``popart::DataTypes`` and
``ONNX_NAMESPACE::TensorProto_DataTypes``, or
``decltype``(ONNX_NAMESPACE::TensorProto().data_type()).)doc";

static const char *__doc_popart_DataType_2 =
    R"doc(There is a one-to-one correspondence between ``popart::DataTypes`` and
``ONNX_NAMESPACE::TensorProto_DataTypes``, or
``decltype``(ONNX_NAMESPACE::TensorProto().data_type()).)doc";

static const char *__doc_popart_DeviceConnectionType = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_2 = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Always = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Always_2 = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Never = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_Never_2 = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_OnDemand = R"doc()doc";

static const char *__doc_popart_DeviceConnectionType_OnDemand_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_2 = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_3 = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_4 = R"doc(Represents a device)doc";

static const char *__doc_popart_DeviceInfo_DeviceInfo = R"doc()doc";

static const char *__doc_popart_DeviceInfo_DeviceInfo_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_attach =
    R"doc(Attach to the device.

Returns:
    True if successfully attached to the device.)doc";

static const char *__doc_popart_DeviceInfo_attach_2 =
    R"doc(Attach to the device.

Returns:
    True if successfully attached to the device.)doc";

static const char *__doc_popart_DeviceInfo_attachTimeout = R"doc()doc";

static const char *__doc_popart_DeviceInfo_attachTimeout_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_canCompileOffline = R"doc()doc";

static const char *__doc_popart_DeviceInfo_canCompileOffline_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_connectionType = R"doc()doc";

static const char *__doc_popart_DeviceInfo_connectionType_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_detach =
    R"doc(Detach from the device.)doc";

static const char *__doc_popart_DeviceInfo_detach_2 =
    R"doc(Detach from the device.)doc";

static const char *__doc_popart_DeviceInfo_flags = R"doc()doc";

static const char *__doc_popart_DeviceInfo_flags_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getConnectionType =
    R"doc(Get the connection type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getConnectionType_2 =
    R"doc(Get the connection type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getDriverIds = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getDriverIds_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getId =
    R"doc(Get the device id.)doc";

static const char *__doc_popart_DeviceInfo_getId_2 =
    R"doc(Get the device id.)doc";

static const char *__doc_popart_DeviceInfo_getNumIpus =
    R"doc(Get the number of IPUs in the device.)doc";

static const char *__doc_popart_DeviceInfo_getNumIpus_2 =
    R"doc(Get the number of IPUs in the device.)doc";

static const char *__doc_popart_DeviceInfo_getNumWorkerContexts =
    R"doc(Get the number of worker contexts per tile.)doc";

static const char *__doc_popart_DeviceInfo_getNumWorkerContexts_2 =
    R"doc(Get the number of worker contexts per tile.)doc";

static const char *__doc_popart_DeviceInfo_getOnDemandAttachTimeout =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_getOnDemandAttachTimeout_2 =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_getOptionFlags = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getOptionFlags_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getTarget = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getTarget_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_getTilesPerIPU =
    R"doc(Get the number of tiles per IPU.)doc";

static const char *__doc_popart_DeviceInfo_getTilesPerIPU_2 =
    R"doc(Get the number of tiles per IPU.)doc";

static const char *__doc_popart_DeviceInfo_getType =
    R"doc(Get the type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getType_2 =
    R"doc(Get the type of the device.)doc";

static const char *__doc_popart_DeviceInfo_getVersion =
    R"doc(Get the version of the software on the IPU.)doc";

static const char *__doc_popart_DeviceInfo_getVersion_2 =
    R"doc(Get the version of the software on the IPU.)doc";

static const char *__doc_popart_DeviceInfo_isAttached =
    R"doc(True if attached.)doc";

static const char *__doc_popart_DeviceInfo_isAttached_2 =
    R"doc(True if attached.)doc";

static const char *__doc_popart_DeviceInfo_provider = R"doc()doc";

static const char *__doc_popart_DeviceInfo_provider_2 = R"doc()doc";

static const char *__doc_popart_DeviceInfo_setOnDemandAttachTimeout =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_setOnDemandAttachTimeout_2 =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_toString =
    R"doc(Return a description of the device.)doc";

static const char *__doc_popart_DeviceInfo_toString_2 =
    R"doc(Return a description of the device.)doc";

static const char *__doc_popart_DeviceInfo_tryAttachUntilTimeout = R"doc()doc";

static const char *__doc_popart_DeviceInfo_tryAttachUntilTimeout_2 =
    R"doc()doc";

static const char *__doc_popart_DeviceInfo_type = R"doc()doc";

static const char *__doc_popart_DeviceInfo_type_2 = R"doc()doc";

static const char *__doc_popart_DeviceManager =
    R"doc(A class to manage devices.)doc";

static const char *__doc_popart_DeviceManager_2 =
    R"doc(A class to manage devices.)doc";

static const char *__doc_popart_DeviceManager_acquireAvailableDevice =
    R"doc(Finds the first available hardware device, with a certain number of
IPUs. This method will attach to the device.

Parameter ``numIpus``:
    The number of IPUs on the device [=1].

Parameter ``tilesPerIPU``:
    The number of tiles per IPU (0 will match any number) [=0]

Returns:
    A device, which can be used with a session. Will return nullptr if
    no device is available.)doc";

static const char *__doc_popart_DeviceManager_acquireAvailableDevice_2 =
    R"doc(Finds the first available hardware device, with a certain number of
IPUs. This method will attach to the device.

Parameter ``numIpus``:
    The number of IPUs on the device [=1].

Parameter ``tilesPerIPU``:
    The number of tiles per IPU (0 will match any number) [=0]

Returns:
    A device, which can be used with a session. Will return nullptr if
    no device is available.)doc";

static const char *__doc_popart_DeviceManager_acquireDeviceById =
    R"doc(Allocates the hardware device by id. This id can be found running `gc-
info -l`. This method will attach to the device.

Parameter ``id``:
    The index of the IPU to be used.

Returns:
    A device. Will return nullptr if the device is not available.)doc";

static const char *__doc_popart_DeviceManager_acquireDeviceById_2 =
    R"doc(Allocates the hardware device by id. This id can be found running `gc-
info -l`. This method will attach to the device.

Parameter ``id``:
    The index of the IPU to be used.

Returns:
    A device. Will return nullptr if the device is not available.)doc";

static const char *__doc_popart_DeviceManager_attachTimeout = R"doc()doc";

static const char *__doc_popart_DeviceManager_attachTimeout_2 = R"doc()doc";

static const char *__doc_popart_DeviceManager_createCpuDevice =
    R"doc(Create a 'simulated' CPU device.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createCpuDevice_2 =
    R"doc(Create a 'simulated' CPU device.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createDeviceManager =
    R"doc(Accessor for the device manager.

Returns:
    A reference to the DeviceManager.)doc";

static const char *__doc_popart_DeviceManager_createDeviceManager_2 =
    R"doc(Accessor for the device manager.

Returns:
    A reference to the DeviceManager.)doc";

static const char *__doc_popart_DeviceManager_createIpuModelDevice =
    R"doc(Create a 'simulated' IPU Model device. The following options are
supported:

* ``numIPUs``: The number of IPUs to simulate [=1] * ``ge``: The
number of tiles per IPU [=defaultFewTiles] * ``compileIPUCode``:
Whether or not to compile real IPU code for modelling

Parameter ``options``:
    Configuration settings for the IPU Model.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createIpuModelDevice_2 =
    R"doc(Create a 'simulated' IPU Model device. The following options are
supported:

* ``numIPUs``: The number of IPUs to simulate [=1] * ``ge``: The
number of tiles per IPU [=defaultFewTiles] * ``compileIPUCode``:
Whether or not to compile real IPU code for modelling

Parameter ``options``:
    Configuration settings for the IPU Model.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createOfflineIPUDevice =
    R"doc(Create a device resembling an IPU for offline compilation, The
following options are supported:

* ``numIPUs``: The number of IPUs to compile for * ``ge``: The number
of tiles per IPU [=defaultManyTiles] * ``ipuVersion``: The ipu
architecture [="ipu1"] * ``syncPattern``: The sync pattern to use:
full/singlePipline/replicaAndLadder, defaults to full

Parameter ``options``:
    Configuration settings for the IPU Model.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createOfflineIPUDevice_2 =
    R"doc(Create a device resembling an IPU for offline compilation, The
following options are supported:

* ``numIPUs``: The number of IPUs to compile for * ``ge``: The number
of tiles per IPU [=defaultManyTiles] * ``ipuVersion``: The ipu
architecture [="ipu1"] * ``syncPattern``: The sync pattern to use:
full/singlePipline/replicaAndLadder, defaults to full

Parameter ``options``:
    Configuration settings for the IPU Model.

Returns:
    A device.)doc";

static const char *__doc_popart_DeviceManager_createSimDevice = R"doc()doc";

static const char *__doc_popart_DeviceManager_createSimDevice_2 = R"doc()doc";

static const char *__doc_popart_DeviceManager_enumerateDevices =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``pattern``:
    Sync pattern.

Parameter ``numIpus``:
    Number of IPUs to request.

Parameter ``deviceType``:
    Type of device required.

Parameter ``tilesPerIPU``:
    The number of tiles per IPU required.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_enumerateDevices_2 =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``pattern``:
    Sync pattern.

Parameter ``numIpus``:
    Number of IPUs to request.

Parameter ``deviceType``:
    Type of device required.

Parameter ``tilesPerIPU``:
    The number of tiles per IPU required.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_getDevice =
    R"doc(Get the Device object of a device by ID.

Parameter ``syncPattern``:
    Sync pattern.

Parameter ``deviceManagerId``:
    Number of IPUs to request.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_getDevice_2 =
    R"doc(Get the Device object of a device by ID.

Parameter ``syncPattern``:
    Sync pattern.

Parameter ``deviceManagerId``:
    Number of IPUs to request.

Returns:
    List of requested IPUs.)doc";

static const char *__doc_popart_DeviceManager_providers = R"doc()doc";

static const char *__doc_popart_DeviceManager_providers_2 = R"doc()doc";

static const char *__doc_popart_DeviceManager_registerDeviceProvider =
    R"doc(Used to register a device provider.

Parameter ``provider``:
    A provider.)doc";

static const char *__doc_popart_DeviceManager_registerDeviceProvider_2 =
    R"doc(Used to register a device provider.

Parameter ``provider``:
    A provider.)doc";

static const char *__doc_popart_DeviceManager_setOnDemandAttachTimeout =
    R"doc(If unable to attach to a device on first try, the attach timeout set
here is the length of time (in seconds) that the DeviceManager will
wait to try and attach. Note: this only takes effect when trying to
attach with a DeviceConnectionType::OnDemand DeviceConnectionType.

Parameter ``seconds``:
    The attach timeout in seconds.)doc";

static const char *__doc_popart_DeviceManager_setOnDemandAttachTimeout_2 =
    R"doc(If unable to attach to a device on first try, the attach timeout set
here is the length of time (in seconds) that the DeviceManager will
wait to try and attach. Note: this only takes effect when trying to
attach with a DeviceConnectionType::OnDemand DeviceConnectionType.

Parameter ``seconds``:
    The attach timeout in seconds.)doc";

static const char *__doc_popart_DeviceProvider =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_2 =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_3 =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_4 =
    R"doc(The interface for device providers which are registered with the
device manager.)doc";

static const char *__doc_popart_DeviceProvider_createHostDevice =
    R"doc(Create a host device for testing.)doc";

static const char *__doc_popart_DeviceProvider_createHostDevice_2 =
    R"doc(Create a host device for testing.)doc";

static const char *__doc_popart_DeviceProvider_enumerate =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``devices``:
    Devices to get.

Parameter ``requiredNumIPUs``:
    Number of IPUs to request.

Parameter ``syncPattern``:
    Sync pattern.

Parameter ``requiredTilesPerIPU``:
    Number of tiles per IPU to request.)doc";

static const char *__doc_popart_DeviceProvider_enumerate_2 =
    R"doc(Get the list of all devices fulfilling the specified criteria.

Parameter ``devices``:
    Devices to get.

Parameter ``requiredNumIPUs``:
    Number of IPUs to request.

Parameter ``syncPattern``:
    Sync pattern.

Parameter ``requiredTilesPerIPU``:
    Number of tiles per IPU to request.)doc";

static const char *__doc_popart_DeviceProvider_getDevice = R"doc()doc";

static const char *__doc_popart_DeviceProvider_getDevice_2 = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_2 = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_First = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_First_2 = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_Random = R"doc()doc";

static const char *__doc_popart_DeviceSelectionCriterion_Random_2 = R"doc()doc";

static const char *__doc_popart_DeviceType = R"doc()doc";

static const char *__doc_popart_DeviceType_2 = R"doc()doc";

static const char *__doc_popart_DeviceType_Cpu = R"doc()doc";

static const char *__doc_popart_DeviceType_Cpu_2 = R"doc()doc";

static const char *__doc_popart_DeviceType_Ipu = R"doc()doc";

static const char *__doc_popart_DeviceType_Ipu_2 = R"doc()doc";

static const char *__doc_popart_DeviceType_IpuModel = R"doc()doc";

static const char *__doc_popart_DeviceType_IpuModel_2 = R"doc()doc";

static const char *__doc_popart_DeviceType_OfflineIpu = R"doc()doc";

static const char *__doc_popart_DeviceType_OfflineIpu_2 = R"doc()doc";

static const char *__doc_popart_DeviceType_Sim = R"doc()doc";

static const char *__doc_popart_DeviceType_Sim_2 = R"doc()doc";

static const char *__doc_popart_DomainOpSet = R"doc()doc";

static const char *__doc_popart_DomainOpSet_2 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet_2 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet_3 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_DomainOpSet_4 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_getOpsetVersion = R"doc()doc";

static const char *__doc_popart_DomainOpSet_getOpsetVersion_2 = R"doc()doc";

static const char *__doc_popart_DomainOpSet_impl = R"doc()doc";

static const char *__doc_popart_DomainOpSet_impl_2 = R"doc()doc";

static const char *__doc_popart_DotCheck =
    R"doc(Enum type used to identify at which stages of IR construction to
export `.dot` files.)doc";

static const char *__doc_popart_DotCheck_2 =
    R"doc(Enum type used to identify at which stages of IR construction to
export `.dot` files.)doc";

static const char *__doc_popart_DotCheck_Bwd0 =
    R"doc(Generate graph after backwards construction.)doc";

static const char *__doc_popart_DotCheck_Bwd0_2 =
    R"doc(Generate graph after backwards construction.)doc";

static const char *__doc_popart_DotCheck_Final =
    R"doc(Generate graph after running aliasing patterns (the final IR).)doc";

static const char *__doc_popart_DotCheck_Final_2 =
    R"doc(Generate graph after running aliasing patterns (the final IR).)doc";

static const char *__doc_popart_DotCheck_Fwd0 =
    R"doc(Generate graph after construction of the forward pass.)doc";

static const char *__doc_popart_DotCheck_Fwd0_2 =
    R"doc(Generate graph after construction of the forward pass.)doc";

static const char *__doc_popart_DotCheck_Fwd1 =
    R"doc(Generate graph after running pre-aliasing patterns.)doc";

static const char *__doc_popart_DotCheck_Fwd1_2 =
    R"doc(Generate graph after running pre-aliasing patterns.)doc";

static const char *__doc_popart_DotCheck_N =
    R"doc(The number of ``DotCheck`` values.)doc";

static const char *__doc_popart_DotCheck_N_2 =
    R"doc(The number of ``DotCheck`` values.)doc";

static const char *__doc_popart_DotCheck_PreAlias =
    R"doc(Generate graph after all transformations, patterns, except the
aliasing.)doc";

static const char *__doc_popart_DotCheck_PreAlias_2 =
    R"doc(Generate graph after all transformations, patterns, except the
aliasing.)doc";

static const char *__doc_popart_ErrorSource = R"doc()doc";

static const char *__doc_popart_ErrorSource_2 = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart_2 = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart_internal = R"doc()doc";

static const char *__doc_popart_ErrorSource_popart_internal_2 = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplar = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplar_2 = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplibs = R"doc()doc";

static const char *__doc_popart_ErrorSource_poplibs_2 = R"doc()doc";

static const char *__doc_popart_ErrorSource_unknown = R"doc()doc";

static const char *__doc_popart_ErrorSource_unknown_2 = R"doc()doc";

static const char *__doc_popart_ExecutionContext = R"doc()doc";

static const char *__doc_popart_ExecutionContext_2 = R"doc()doc";

static const char *__doc_popart_ExecutionContext_AccumulateOuterFragment =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_AccumulateOuterFragment_2 =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_Normal = R"doc()doc";

static const char *__doc_popart_ExecutionContext_Normal_2 = R"doc()doc";

static const char *__doc_popart_ExecutionContext_OptimizerFromHostFragment =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_OptimizerFromHostFragment_2 =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_Subgraph = R"doc()doc";

static const char *__doc_popart_ExecutionContext_Subgraph_2 = R"doc()doc";

static const char *__doc_popart_ExecutionContext_WeightsFromHostFragment =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_WeightsFromHostFragment_2 =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_WeightsToHostFragment =
    R"doc()doc";

static const char *__doc_popart_ExecutionContext_WeightsToHostFragment_2 =
    R"doc()doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule =
    R"doc(Enum type to specify when to load tensors.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_2 =
    R"doc(Enum type to specify when to load tensors.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_N =
    R"doc(The number of ``ExecutionPhaseIOSchedule`` values.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_N_2 =
    R"doc(The number of ``ExecutionPhaseIOSchedule`` values.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_OnDemand =
    R"doc(Load tensors just before they are required.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_OnDemand_2 =
    R"doc(Load tensors just before they are required.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_Preload =
    R"doc(Preload tensors in previous phase for use in current phase.)doc";

static const char *__doc_popart_ExecutionPhaseIOSchedule_Preload_2 =
    R"doc(Preload tensors in previous phase for use in current phase.)doc";

static const char *__doc_popart_ExecutionPhaseSchedule =
    R"doc(Enum type to specify the order of processing optimizer operations for
different weights of the same execution phase.

The steps for phased execution consists of: - Copy to IO tiles if
necessary (1) - Run collective operations if necessary (2) - Load
optimizer state (3) - Update optimizer state (4) - Apply optimizer (5)
- Store updated tensor if necessary (6))doc";

static const char *__doc_popart_ExecutionPhaseSchedule_2 =
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

static const char *__doc_popart_ExecutionPhaseSchedule_Batch_2 =
    R"doc(Process above steps for all weights together, in a way that maximises
overlap potential between compute and exchange (for example: 333, 111,
222, 444, 555, 666).)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_BatchClusteredIO =
    R"doc(Process above steps for all weights together, in a way that maximises
overlap potential between compute and exchange, and maximise stream
copy merges by keeping RemoteLoad/RemoteStore operations clustered
(for example: 333, 111, 222, 444, 555, 666).)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_BatchClusteredIO_2 =
    R"doc(Process above steps for all weights together, in a way that maximises
overlap potential between compute and exchange, and maximise stream
copy merges by keeping RemoteLoad/RemoteStore operations clustered
(for example: 333, 111, 222, 444, 555, 666).)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_Interleaving =
    R"doc(Process above steps for one weight at a time (for example: 123456,
123456, 123456). The scheduler may interleave these steps.)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_Interleaving_2 =
    R"doc(Process above steps for one weight at a time (for example: 123456,
123456, 123456). The scheduler may interleave these steps.)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_N =
    R"doc(The number of ``ExecutionPhaseSchedule`` values.)doc";

static const char *__doc_popart_ExecutionPhaseSchedule_N_2 =
    R"doc(The number of ``ExecutionPhaseSchedule`` values.)doc";

static const char *__doc_popart_ExecutionPhaseSettings =
    R"doc(A structure containing ExecutionPhase settings.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_2 =
    R"doc(A structure containing ExecutionPhase settings.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings =
    R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings_2 = R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings_3 = R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_ExecutionPhaseSettings_4 = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_accumulatorIOSchedule =
    R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_accumulatorIOSchedule_2 =
    R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_activationIOSchedule =
    R"doc(The execution phase IO schedule for activation and gradient tensors.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_activationIOSchedule_2 =
    R"doc(The execution phase IO schedule for activation and gradient tensors.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_operator_assign =
    R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_operator_assign_2 =
    R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_optimizerStateIOSchedule = R"doc()doc";

static const char
    *__doc_popart_ExecutionPhaseSettings_optimizerStateIOSchedule_2 =
        R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_phases =
    R"doc(Number of ExecutionPhases for the whole model)doc";

static const char *__doc_popart_ExecutionPhaseSettings_phases_2 =
    R"doc(Number of ExecutionPhases for the whole model)doc";

static const char *__doc_popart_ExecutionPhaseSettings_schedule = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_schedule_2 = R"doc()doc";

static const char *__doc_popart_ExecutionPhaseSettings_stages =
    R"doc(Number of overlapping stages 1: Parallel streaming memory, default for
1 IPU / replica 2: PingPong between 2 IPUs, default for >= 2 IPUs /
replica)doc";

static const char *__doc_popart_ExecutionPhaseSettings_stages_2 =
    R"doc(Number of overlapping stages 1: Parallel streaming memory, default for
1 IPU / replica 2: PingPong between 2 IPUs, default for >= 2 IPUs /
replica)doc";

static const char *__doc_popart_ExecutionPhaseSettings_weightIOSchedule =
    R"doc(The execution phase IO schedule for weight tensors.)doc";

static const char *__doc_popart_ExecutionPhaseSettings_weightIOSchedule_2 =
    R"doc(The execution phase IO schedule for weight tensors.)doc";

static const char *__doc_popart_GradInOutMapper = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_2 = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_GradInOutMapper = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_GradInOutMapper_2 = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_iGrad = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_iGrad_2 = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_iNonGrad = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_iNonGrad_2 = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_operator_eq = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_operator_eq_2 = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_type = R"doc()doc";

static const char *__doc_popart_GradInOutMapper_type_2 = R"doc()doc";

static const char *__doc_popart_GradOpInType =
    R"doc(The relationship between the input tensor of a grad-op and the
corresponding non-grad-op.)doc";

static const char *__doc_popart_GradOpInType_2 =
    R"doc(The relationship between the input tensor of a grad-op and the
corresponding non-grad-op.)doc";

static const char *__doc_popart_GradOpInType_GradOut = R"doc()doc";

static const char *__doc_popart_GradOpInType_GradOut_2 = R"doc()doc";

static const char *__doc_popart_GradOpInType_In = R"doc()doc";

static const char *__doc_popart_GradOpInType_In_2 = R"doc()doc";

static const char *__doc_popart_GradOpInType_Out = R"doc()doc";

static const char *__doc_popart_GradOpInType_Out_2 = R"doc()doc";

static const char *__doc_popart_GraphTransformer = R"doc()doc";

static const char *__doc_popart_GraphTransformer_2 = R"doc()doc";

static const char *__doc_popart_GraphTransformerImpl = R"doc()doc";

static const char *__doc_popart_GraphTransformerImpl_2 = R"doc()doc";

static const char *__doc_popart_GraphTransformer_GraphTransformer = R"doc()doc";

static const char *__doc_popart_GraphTransformer_GraphTransformer_2 =
    R"doc()doc";

static const char
    *__doc_popart_GraphTransformer_convertAllFixedPointInitializersToConstants =
        R"doc(Convert all of the fixed-point initializers into ONNX Constant Nodes)doc";

static const char *
    __doc_popart_GraphTransformer_convertAllFixedPointInitializersToConstants_2 =
        R"doc(Convert all of the fixed-point initializers into ONNX Constant Nodes)doc";

static const char *__doc_popart_GraphTransformer_convertBFloats16ToFloat32 =
    R"doc(Convert the graph from BFloat16 to Float32)doc";

static const char *__doc_popart_GraphTransformer_convertBFloats16ToFloat32_2 =
    R"doc(Convert the graph from BFloat16 to Float32)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToFloats =
    R"doc(Convert the graph from float64 to float32)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToFloats_2 =
    R"doc(Convert the graph from float64 to float32)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToHalfs =
    R"doc(Convert the graph from float64 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertDoublesToHalfs_2 =
    R"doc(Convert the graph from float64 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertFloatsToHalfs =
    R"doc(Convert the graph from float32 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertFloatsToHalfs_2 =
    R"doc(Convert the graph from float32 to float16)doc";

static const char *__doc_popart_GraphTransformer_convertINT16ToINT32 =
    R"doc(Convert the graph from int16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertINT16ToINT32_2 =
    R"doc(Convert the graph from int16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertINT64ToINT32 =
    R"doc(Convert the graph from int64 to int32

Parameter ``clip``:
    If tensor data are outside of the numerical range expressible by
    int32, clip to max and min numeric limits)doc";

static const char *__doc_popart_GraphTransformer_convertINT64ToINT32_2 =
    R"doc(Convert the graph from int64 to int32

Parameter ``clip``:
    If tensor data are outside of the numerical range expressible by
    int32, clip to max and min numeric limits)doc";

static const char *__doc_popart_GraphTransformer_convertINT8ToINT32 =
    R"doc(Convert the graph from int8 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertINT8ToINT32_2 =
    R"doc(Convert the graph from int8 to int32)doc";

static const char
    *__doc_popart_GraphTransformer_convertInitializersToConstants =
        R"doc(Convert the given list of initializers into ONNX Constant Nodes

Parameter ``ids``:
    A list of initializer names)doc";

static const char
    *__doc_popart_GraphTransformer_convertInitializersToConstants_2 =
        R"doc(Convert the given list of initializers into ONNX Constant Nodes

Parameter ``ids``:
    A list of initializer names)doc";

static const char *__doc_popart_GraphTransformer_convertUINT16ToINT32 =
    R"doc(Convert the graph from uint16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertUINT16ToINT32_2 =
    R"doc(Convert the graph from uint16 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertUINT8ToINT32 =
    R"doc(Convert the graph from uint8 to int32)doc";

static const char *__doc_popart_GraphTransformer_convertUINT8ToINT32_2 =
    R"doc(Convert the graph from uint8 to int32)doc";

static const char *__doc_popart_GraphTransformer_getModelProto = R"doc()doc";

static const char *__doc_popart_GraphTransformer_getModelProto_2 = R"doc()doc";

static const char *__doc_popart_GraphTransformer_impl = R"doc()doc";

static const char *__doc_popart_GraphTransformer_impl_2 = R"doc()doc";

static const char *__doc_popart_GraphTransformer_prepareNodesForTraining =
    R"doc(Some ONNX Operators are different between train and test modes An
example is BatchNormalization, which has 1 output in test mode and 5
outputs in train mode This function changes the Nodes to be of the
training variety)doc";

static const char *__doc_popart_GraphTransformer_prepareNodesForTraining_2 =
    R"doc(Some ONNX Operators are different between train and test modes An
example is BatchNormalization, which has 1 output in test mode and 5
outputs in train mode This function changes the Nodes to be of the
training variety)doc";

static const char *__doc_popart_GraphTransformer_removeUnusedInputs =
    R"doc(Inputs which are not connected to any Node are removed)doc";

static const char *__doc_popart_GraphTransformer_removeUnusedInputs_2 =
    R"doc(Inputs which are not connected to any Node are removed)doc";

static const char *__doc_popart_GraphTransformer_saveInitializersExternally =
    R"doc(The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To prevent this for large models, ONNX tensor data can be
saved separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data.)doc";

static const char *__doc_popart_GraphTransformer_saveInitializersExternally_2 =
    R"doc(The model data cannot exceed 2GB - the maximum size of a Protobuf
message. To prevent this for large models, ONNX tensor data can be
saved separately.

Parameter ``ids``:
    The names of tensors whose data is to be saved externally.

Parameter ``fn``:
    The name of a file containing the binary tensor data.)doc";

static const char *__doc_popart_IdentityGradOp = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp_3 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_IdentityGradOp_4 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_clone_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getInIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradInputInfo_2 = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_IdentityGradOp_gradOutToNonGradIn_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp_3 =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_IdentityInplaceOp_4 =
    R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_aliases = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_aliases_2 = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityInplaceOp_clone_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_IdentityLossGradOp =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_IdentityLossGradOp_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_canBeReplacedByIdentity =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_canBeReplacedByIdentity_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_canShard = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_clone_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getInIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getReductionType =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getReductionType_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getScaleByReplication =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getScaleByReplication_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getShardRescaleFactor =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getShardRescaleFactor_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getSubgraphValue =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_getSubgraphValue_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradInputInfo_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradOutToNonGradIn =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_gradOutToNonGradIn_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_outShape = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_outShape_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_reduction_type = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_reduction_type_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_scaleByReplication =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_scaleByReplication_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_setup = R"doc()doc";

static const char *__doc_popart_IdentityLossGradOp_setup_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_IdentityLossOp = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_IdentityLossOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_canBeReplacedByIdentity =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_canBeReplacedByIdentity_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_canShard = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_clone_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getGradOps = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getGradOps_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getInIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getInIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getScaleByReplicationOverride =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getScaleByReplicationOverride_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getShardReductionType =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getShardReductionType_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_scaleByReplicationOverride =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_scaleByReplicationOverride_2 =
    R"doc()doc";

static const char *__doc_popart_IdentityLossOp_setup = R"doc()doc";

static const char *__doc_popart_IdentityLossOp_setup_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp = R"doc()doc";

static const char *__doc_popart_IdentityOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp_IdentityOp = R"doc()doc";

static const char *__doc_popart_IdentityOp_IdentityOp_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp_clone = R"doc()doc";

static const char *__doc_popart_IdentityOp_clone_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp_getGradOps = R"doc()doc";

static const char *__doc_popart_IdentityOp_getGradOps_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp_getInplaceVariant = R"doc()doc";

static const char *__doc_popart_IdentityOp_getInplaceVariant_2 = R"doc()doc";

static const char *__doc_popart_IdentityOp_inplacePriorityDefault = R"doc()doc";

static const char *__doc_popart_IdentityOp_inplacePriorityDefault_2 =
    R"doc()doc";

static const char *__doc_popart_InferenceSession = R"doc()doc";

static const char *__doc_popart_InferenceSession_2 = R"doc()doc";

static const char *__doc_popart_InferenceSession_createFromIr = R"doc()doc";

static const char *__doc_popart_InferenceSession_createFromIr_2 = R"doc()doc";

static const char *__doc_popart_InferenceSession_createFromOnnxModel =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for inference.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors.

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches.

Parameter ``userOptions``:
    String to configure session options.

Parameter ``patterns``:
    Optimization patterns to apply.)doc";

static const char *__doc_popart_InferenceSession_createFromOnnxModel_2 =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for inference.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors.

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches.

Parameter ``userOptions``:
    String to configure session options.

Parameter ``patterns``:
    Optimization patterns to apply.)doc";

static const char *__doc_popart_InitOp = R"doc()doc";

static const char *__doc_popart_InitOp_2 = R"doc()doc";

static const char *__doc_popart_InitOp_InitOp = R"doc()doc";

static const char *__doc_popart_InitOp_InitOp_2 = R"doc()doc";

static const char *__doc_popart_InitOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_InitOp_appendOutlineAttributes_2 = R"doc()doc";

static const char *__doc_popart_InitOp_batch_axis = R"doc()doc";

static const char *__doc_popart_InitOp_batch_axis_2 = R"doc()doc";

static const char *__doc_popart_InitOp_canShard = R"doc()doc";

static const char *__doc_popart_InitOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_InitOp_clone = R"doc()doc";

static const char *__doc_popart_InitOp_clone_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getInitType = R"doc()doc";

static const char *__doc_popart_InitOp_getInitType_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getOutBatchAxis = R"doc()doc";

static const char *__doc_popart_InitOp_getOutBatchAxis_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_InitOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_InitOp_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorInfo = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorInfo_2 = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorType = R"doc()doc";

static const char *__doc_popart_InitOp_getTensorType_2 = R"doc()doc";

static const char *__doc_popart_InitOp_init_type = R"doc()doc";

static const char *__doc_popart_InitOp_init_type_2 = R"doc()doc";

static const char *__doc_popart_InitOp_isOutlineable = R"doc()doc";

static const char *__doc_popart_InitOp_isOutlineable_2 = R"doc()doc";

static const char *__doc_popart_InitOp_setup = R"doc()doc";

static const char *__doc_popart_InitOp_setup_2 = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_info = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_info_2 = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_type = R"doc()doc";

static const char *__doc_popart_InitOp_tensor_type_2 = R"doc()doc";

static const char *__doc_popart_InitType = R"doc()doc";

static const char *__doc_popart_InitType_2 = R"doc()doc";

static const char *__doc_popart_InitType_NoInit = R"doc()doc";

static const char *__doc_popart_InitType_NoInit_2 = R"doc()doc";

static const char *__doc_popart_InitType_Zero = R"doc()doc";

static const char *__doc_popart_InitType_Zero_2 = R"doc()doc";

static const char *__doc_popart_Instrumentation =
    R"doc(Enum type used to specify an instrumentation type.)doc";

static const char *__doc_popart_Instrumentation_2 =
    R"doc(Enum type used to specify an instrumentation type.)doc";

static const char *__doc_popart_Instrumentation_Inner =
    R"doc(Inner loop instrumentation, graph per IPU.)doc";

static const char *__doc_popart_Instrumentation_Inner_2 =
    R"doc(Inner loop instrumentation, graph per IPU.)doc";

static const char *__doc_popart_Instrumentation_N =
    R"doc(The number of ``Instrumentation`` values.)doc";

static const char *__doc_popart_Instrumentation_N_2 =
    R"doc(The number of ``Instrumentation`` values.)doc";

static const char *__doc_popart_Instrumentation_Outer =
    R"doc(Outer loop instrumentation, graph over all IPUs.)doc";

static const char *__doc_popart_Instrumentation_Outer_2 =
    R"doc(Outer loop instrumentation, graph over all IPUs.)doc";

static const char *__doc_popart_Ir = R"doc()doc";

static const char *__doc_popart_Ir_2 = R"doc()doc";

static const char *__doc_popart_IrBundle = R"doc()doc";

static const char *__doc_popart_IrBundle_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_IrBundle = R"doc()doc";

static const char *__doc_popart_IrBundle_IrBundle_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_dataFlow = R"doc()doc";

static const char *__doc_popart_IrBundle_dataFlow_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_deviceInfo = R"doc()doc";

static const char *__doc_popart_IrBundle_deviceInfo_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_inputShapeInfo = R"doc()doc";

static const char *__doc_popart_IrBundle_inputShapeInfo_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_loss = R"doc()doc";

static const char *__doc_popart_IrBundle_loss_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_modelProto = R"doc()doc";

static const char *__doc_popart_IrBundle_modelProto_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_optimizer = R"doc()doc";

static const char *__doc_popart_IrBundle_optimizer_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_patterns = R"doc()doc";

static const char *__doc_popart_IrBundle_patterns_2 = R"doc()doc";

static const char *__doc_popart_IrBundle_userOptions = R"doc()doc";

static const char *__doc_popart_IrBundle_userOptions_2 = R"doc()doc";

static const char *__doc_popart_IrSerializationFormat =
    R"doc(Enum type used to specify a serialization format.)doc";

static const char *__doc_popart_IrSerializationFormat_2 =
    R"doc(Enum type used to specify a serialization format.)doc";

static const char *__doc_popart_IrSerializationFormat_JSON =
    R"doc(JavaScript Object Notation (JSON).)doc";

static const char *__doc_popart_IrSerializationFormat_JSON_2 =
    R"doc(JavaScript Object Notation (JSON).)doc";

static const char *__doc_popart_Ir_ExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_2 = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Inference = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Inference_2 = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Training = R"doc()doc";

static const char *__doc_popart_Ir_ExecutionMode_Training_2 = R"doc()doc";

static const char *__doc_popart_Ir_Ir = R"doc()doc";

static const char *__doc_popart_Ir_Ir_2 = R"doc()doc";

static const char *__doc_popart_Ir_Ir_3 = R"doc()doc";

static const char *__doc_popart_Ir_Ir_4 = R"doc()doc";

static const char *__doc_popart_Ir_Ir_5 = R"doc()doc";

static const char *__doc_popart_Ir_Ir_6 = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat_2 = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat_JSON = R"doc()doc";

static const char *__doc_popart_Ir_SerialiseFormat_JSON_2 = R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor = R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor_3 =
    R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensor_4 =
    R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_addAdditionalModelProtoTensors_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_addOp = R"doc()doc";

static const char *__doc_popart_Ir_addOp_2 = R"doc()doc";

static const char *__doc_popart_Ir_additionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_additionalModelProtoTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_append = R"doc()doc";

static const char *__doc_popart_Ir_append_2 = R"doc()doc";

static const char *__doc_popart_Ir_applyInplacePattern = R"doc()doc";

static const char *__doc_popart_Ir_applyInplacePattern_2 = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPattern = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPattern_2 = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPatterns = R"doc()doc";

static const char *__doc_popart_Ir_applyPreAliasPatterns_2 = R"doc()doc";

static const char *__doc_popart_Ir_applyTransform = R"doc()doc";

static const char *__doc_popart_Ir_applyTransform_2 = R"doc()doc";

static const char *__doc_popart_Ir_applyUpdateInplacePrioritiesForIpu =
    R"doc()doc";

static const char *__doc_popart_Ir_applyUpdateInplacePrioritiesForIpu_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_autoRecomputationEnabled = R"doc()doc";

static const char *__doc_popart_Ir_autoRecomputationEnabled_2 = R"doc()doc";

static const char *__doc_popart_Ir_canInfer = R"doc()doc";

static const char *__doc_popart_Ir_canInfer_2 = R"doc()doc";

static const char *__doc_popart_Ir_canTrain = R"doc()doc";

static const char *__doc_popart_Ir_canTrain_2 = R"doc()doc";

static const char *__doc_popart_Ir_compareWithSavedHash = R"doc()doc";

static const char *__doc_popart_Ir_compareWithSavedHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_computeHash = R"doc()doc";

static const char *__doc_popart_Ir_computeHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_confirmConstIds = R"doc()doc";

static const char *__doc_popart_Ir_confirmConstIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_confirmNoReservedIds = R"doc()doc";

static const char *__doc_popart_Ir_confirmNoReservedIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_confirmNonReservedId = R"doc()doc";

static const char *__doc_popart_Ir_confirmNonReservedId_2 = R"doc()doc";

static const char *__doc_popart_Ir_constructBackwards = R"doc()doc";

static const char *__doc_popart_Ir_constructBackwards_2 = R"doc()doc";

static const char *__doc_popart_Ir_constructForwards = R"doc()doc";

static const char *__doc_popart_Ir_constructForwards_2 = R"doc()doc";

static const char *__doc_popart_Ir_constructFromOnnxGraph = R"doc()doc";

static const char *__doc_popart_Ir_constructFromOnnxGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_constructedBackwards = R"doc()doc";

static const char *__doc_popart_Ir_constructedBackwards_2 = R"doc()doc";

static const char *__doc_popart_Ir_constructedFinalLoss = R"doc()doc";

static const char *__doc_popart_Ir_constructedFinalLoss_2 = R"doc()doc";

static const char *__doc_popart_Ir_containsInitialisers = R"doc()doc";

static const char *__doc_popart_Ir_containsInitialisers_2 = R"doc()doc";

static const char *__doc_popart_Ir_containsTensor = R"doc()doc";

static const char *__doc_popart_Ir_containsTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_createConcatTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createConcatTensorId_2 = R"doc()doc";

static const char *__doc_popart_Ir_createGraph = R"doc()doc";

static const char *__doc_popart_Ir_createGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_createIntermediateTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createIntermediateTensorId_2 = R"doc()doc";

static const char *__doc_popart_Ir_createSliceTensorId = R"doc()doc";

static const char *__doc_popart_Ir_createSliceTensorId_2 = R"doc()doc";

static const char *__doc_popart_Ir_createUniqueSubgraphId = R"doc()doc";

static const char *__doc_popart_Ir_createUniqueSubgraphId_2 = R"doc()doc";

static const char *__doc_popart_Ir_dataFlow = R"doc()doc";

static const char *__doc_popart_Ir_dataFlow_2 = R"doc()doc";

static const char *__doc_popart_Ir_dataStreamTensors = R"doc()doc";

static const char *__doc_popart_Ir_dataStreamTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_decomposedOptimizers = R"doc()doc";

static const char *__doc_popart_Ir_decomposedOptimizers_2 = R"doc()doc";

static const char *__doc_popart_Ir_deviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_deviceInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_dotCheckpoint = R"doc()doc";

static const char *__doc_popart_Ir_dotCheckpoint_2 = R"doc()doc";

static const char *__doc_popart_Ir_enableTransform = R"doc()doc";

static const char *__doc_popart_Ir_enableTransform_2 = R"doc()doc";

static const char *__doc_popart_Ir_ensureOptimizerTensorCreated = R"doc()doc";

static const char *__doc_popart_Ir_ensureOptimizerTensorCreated_2 = R"doc()doc";

static const char *__doc_popart_Ir_executionMode = R"doc()doc";

static const char *__doc_popart_Ir_executionMode_2 = R"doc()doc";

static const char *__doc_popart_Ir_executionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_executionPhasesReady_2 = R"doc()doc";

static const char *__doc_popart_Ir_finalLossId = R"doc()doc";

static const char *__doc_popart_Ir_finalLossId_2 = R"doc()doc";

static const char *__doc_popart_Ir_finalLossOpId = R"doc()doc";

static const char *__doc_popart_Ir_finalLossOpId_2 = R"doc()doc";

static const char *__doc_popart_Ir_finalizeOpDebugInfo = R"doc()doc";

static const char *__doc_popart_Ir_finalizeOpDebugInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_foldConstants = R"doc()doc";

static const char *__doc_popart_Ir_foldConstants_2 = R"doc()doc";

static const char *__doc_popart_Ir_getAccumulateOuterFragmentBinConstraints =
    R"doc()doc";

static const char *__doc_popart_Ir_getAccumulateOuterFragmentBinConstraints_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors = R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors_3 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAdditionalModelProtoTensors_4 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAllGraphs = R"doc()doc";

static const char *__doc_popart_Ir_getAllGraphs_2 = R"doc()doc";

static const char *__doc_popart_Ir_getAllOps = R"doc()doc";

static const char *__doc_popart_Ir_getAllOps_2 = R"doc()doc";

static const char *__doc_popart_Ir_getAllRemoteBufferInfos = R"doc()doc";

static const char *__doc_popart_Ir_getAllRemoteBufferInfos_2 = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrOpsCounter = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrOpsCounter_2 = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementRandomReferenceId =
    R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementRandomReferenceId_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementSeedModifier = R"doc()doc";

static const char *__doc_popart_Ir_getAndIncrementSeedModifier_2 = R"doc()doc";

static const char *__doc_popart_Ir_getDataFlow = R"doc()doc";

static const char *__doc_popart_Ir_getDataFlow_2 = R"doc()doc";

static const char *__doc_popart_Ir_getDefaultOpsetVersion = R"doc()doc";

static const char *__doc_popart_Ir_getDefaultOpsetVersion_2 = R"doc()doc";

static const char *__doc_popart_Ir_getDeviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_getDeviceInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionMode_2 = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_getExecutionPhasesReady_2 = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossId = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossId_2 = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossOpId = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossOpId_2 = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossPipelineStage = R"doc()doc";

static const char *__doc_popart_Ir_getFinalLossPipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Ir_getGraph = R"doc()doc";

static const char *__doc_popart_Ir_getGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_getGraphInputIds = R"doc()doc";

static const char *__doc_popart_Ir_getGraphInputIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_getGraphSchedule = R"doc()doc";

static const char *__doc_popart_Ir_getGraphSchedule_2 = R"doc()doc";

static const char *__doc_popart_Ir_getGraphs = R"doc()doc";

static const char *__doc_popart_Ir_getGraphs_2 = R"doc()doc";

static const char *__doc_popart_Ir_getHash = R"doc()doc";

static const char *__doc_popart_Ir_getHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_getInputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_getInputShapeInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_getIrBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_getIrBundleHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph_3 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraph_4 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps_3 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphOps_4 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors_3 = R"doc()doc";

static const char *__doc_popart_Ir_getMainGraphTensors_4 = R"doc()doc";

static const char *__doc_popart_Ir_getMaxVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Ir_getMaxVirtualGraphId_2 = R"doc()doc";

static const char *__doc_popart_Ir_getModel = R"doc()doc";

static const char *__doc_popart_Ir_getModel_2 = R"doc()doc";

static const char *__doc_popart_Ir_getModelInputIds = R"doc()doc";

static const char *__doc_popart_Ir_getModelInputIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_getNumPipelineStages = R"doc()doc";

static const char *__doc_popart_Ir_getNumPipelineStages_2 = R"doc()doc";

static const char *__doc_popart_Ir_getOpSchedule = R"doc()doc";

static const char *__doc_popart_Ir_getOpSchedule_2 = R"doc()doc";

static const char *__doc_popart_Ir_getOpSetVersionFromModel = R"doc()doc";

static const char *__doc_popart_Ir_getOpSetVersionFromModel_2 = R"doc()doc";

static const char *__doc_popart_Ir_getOpsCounter = R"doc()doc";

static const char *__doc_popart_Ir_getOpsCounter_2 = R"doc()doc";

static const char *__doc_popart_Ir_getOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_getOptimizer_2 = R"doc()doc";

static const char *__doc_popart_Ir_getOrSetRandomReferenceTensor = R"doc()doc";

static const char *__doc_popart_Ir_getOrSetRandomReferenceTensor_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_getPatternLevelStr = R"doc()doc";

static const char *__doc_popart_Ir_getPatternLevelStr_2 = R"doc()doc";

static const char *__doc_popart_Ir_getPatterns = R"doc()doc";

static const char *__doc_popart_Ir_getPatterns_2 = R"doc()doc";

static const char *__doc_popart_Ir_getRemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_Ir_getRemoteBufferInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_getRequiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_getRequiresRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Ir_getRootInputsToOp = R"doc()doc";

static const char *__doc_popart_Ir_getRootInputsToOp_2 = R"doc()doc";

static const char *__doc_popart_Ir_getSessionOptions = R"doc()doc";

static const char *__doc_popart_Ir_getSessionOptions_2 = R"doc()doc";

static const char *__doc_popart_Ir_getSubgraphAnchorPlaceholder = R"doc()doc";

static const char *__doc_popart_Ir_getSubgraphAnchorPlaceholder_2 = R"doc()doc";

static const char *__doc_popart_Ir_getTensor = R"doc()doc";

static const char *__doc_popart_Ir_getTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_getTensorIds = R"doc()doc";

static const char *__doc_popart_Ir_getTensorIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_getTensors = R"doc()doc";

static const char *__doc_popart_Ir_getTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_getTensors_3 = R"doc()doc";

static const char *__doc_popart_Ir_getTensors_4 = R"doc()doc";

static const char *__doc_popart_Ir_getTrainTargetOps = R"doc()doc";

static const char *__doc_popart_Ir_getTrainTargetOps_2 = R"doc()doc";

static const char *__doc_popart_Ir_getVirtualGraphIdFromTensorProducers =
    R"doc()doc";

static const char *__doc_popart_Ir_getVirtualGraphIdFromTensorProducers_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_graphs = R"doc()doc";

static const char *__doc_popart_Ir_graphs_2 = R"doc()doc";

static const char *__doc_popart_Ir_growCopyVarUpdateOp = R"doc()doc";

static const char *__doc_popart_Ir_growCopyVarUpdateOp_2 = R"doc()doc";

static const char *__doc_popart_Ir_growGradientVarUpdateOp = R"doc()doc";

static const char *__doc_popart_Ir_growGradientVarUpdateOp_2 = R"doc()doc";

static const char *__doc_popart_Ir_growVarUpdateOpInternal = R"doc()doc";

static const char *__doc_popart_Ir_growVarUpdateOpInternal_2 = R"doc()doc";

static const char *__doc_popart_Ir_hasConstructedBackwards = R"doc()doc";

static const char *__doc_popart_Ir_hasConstructedBackwards_2 = R"doc()doc";

static const char *__doc_popart_Ir_hasDecomposedOptimizers = R"doc()doc";

static const char *__doc_popart_Ir_hasDecomposedOptimizers_2 = R"doc()doc";

static const char *__doc_popart_Ir_hasGraph = R"doc()doc";

static const char *__doc_popart_Ir_hasGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_hasRandomOps = R"doc()doc";

static const char *__doc_popart_Ir_hasRandomOps_2 = R"doc()doc";

static const char *__doc_popart_Ir_hasReplicatedTensorSharding = R"doc()doc";

static const char *__doc_popart_Ir_hasReplicatedTensorSharding_2 = R"doc()doc";

static const char *__doc_popart_Ir_hash = R"doc()doc";

static const char *__doc_popart_Ir_hash_2 = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched_2 = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched_3 = R"doc()doc";

static const char *__doc_popart_Ir_hashMatched_4 = R"doc()doc";

static const char *__doc_popart_Ir_initRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_initRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Ir_inputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_inputShapeInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_intermediate_tensor_counter = R"doc()doc";

static const char *__doc_popart_Ir_intermediate_tensor_counter_2 = R"doc()doc";

static const char *__doc_popart_Ir_irBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_irBundleHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_isAnchored = R"doc()doc";

static const char *__doc_popart_Ir_isAnchored_2 = R"doc()doc";

static const char *__doc_popart_Ir_isCandidateForConstExprFolding = R"doc()doc";

static const char *__doc_popart_Ir_isCandidateForConstExprFolding_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_isConsumedByOpOfType = R"doc()doc";

static const char *__doc_popart_Ir_isConsumedByOpOfType_2 = R"doc()doc";

static const char *__doc_popart_Ir_isPatternsLevel = R"doc()doc";

static const char *__doc_popart_Ir_isPatternsLevel_2 = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared_2 = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared_3 = R"doc()doc";

static const char *__doc_popart_Ir_isPrepared_4 = R"doc()doc";

static const char *__doc_popart_Ir_isSchedulable = R"doc()doc";

static const char *__doc_popart_Ir_isSchedulable_2 = R"doc()doc";

static const char *__doc_popart_Ir_isTesting = R"doc()doc";

static const char *__doc_popart_Ir_isTesting_2 = R"doc()doc";

static const char *__doc_popart_Ir_isTraining = R"doc()doc";

static const char *__doc_popart_Ir_isTraining_2 = R"doc()doc";

static const char *__doc_popart_Ir_logIr = R"doc()doc";

static const char *__doc_popart_Ir_logIr_2 = R"doc()doc";

static const char *__doc_popart_Ir_mergeRandomReferenceIds = R"doc()doc";

static const char *__doc_popart_Ir_mergeRandomReferenceIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_onnxModel = R"doc()doc";

static const char *__doc_popart_Ir_onnxModel_2 = R"doc()doc";

static const char *__doc_popart_Ir_opAndRootInputs = R"doc()doc";

static const char *__doc_popart_Ir_opAndRootInputs_2 = R"doc()doc";

static const char *__doc_popart_Ir_operator_assign = R"doc()doc";

static const char *__doc_popart_Ir_operator_assign_2 = R"doc()doc";

static const char *__doc_popart_Ir_operator_assign_3 = R"doc()doc";

static const char *__doc_popart_Ir_operator_assign_4 = R"doc()doc";

static const char *__doc_popart_Ir_opsCounter = R"doc()doc";

static const char *__doc_popart_Ir_opsCounter_2 = R"doc()doc";

static const char *__doc_popart_Ir_opsOfType = R"doc()doc";

static const char *__doc_popart_Ir_opsOfType_2 = R"doc()doc";

static const char *__doc_popart_Ir_optimizer = R"doc()doc";

static const char *__doc_popart_Ir_optimizer_2 = R"doc()doc";

static const char *__doc_popart_Ir_optimizerTensors = R"doc()doc";

static const char *__doc_popart_Ir_optimizerTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_patterns = R"doc()doc";

static const char *__doc_popart_Ir_patterns_2 = R"doc()doc";

static const char *__doc_popart_Ir_prepare = R"doc()doc";

static const char *__doc_popart_Ir_prepare_2 = R"doc()doc";

static const char *__doc_popart_Ir_prepareImpl = R"doc()doc";

static const char *__doc_popart_Ir_prepareImpl_2 = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceId = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceId_2 = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceTensorMap = R"doc()doc";

static const char *__doc_popart_Ir_randomReferenceTensorMap_2 = R"doc()doc";

static const char *__doc_popart_Ir_registerInputTensors = R"doc()doc";

static const char *__doc_popart_Ir_registerInputTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_remoteBufferInfoMap = R"doc()doc";

static const char *__doc_popart_Ir_remoteBufferInfoMap_2 = R"doc()doc";

static const char *__doc_popart_Ir_removeGraph = R"doc()doc";

static const char *__doc_popart_Ir_removeGraph_2 = R"doc()doc";

static const char *__doc_popart_Ir_removeIsolatedTensors = R"doc()doc";

static const char *__doc_popart_Ir_removeIsolatedTensors_2 = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed_3 = R"doc()doc";

static const char *__doc_popart_Ir_requiresRandomSeed_4 = R"doc()doc";

static const char *__doc_popart_Ir_seedModifier = R"doc()doc";

static const char *__doc_popart_Ir_seedModifier_2 = R"doc()doc";

static const char *__doc_popart_Ir_serialise = R"doc()doc";

static const char *__doc_popart_Ir_serialise_2 = R"doc()doc";

static const char *__doc_popart_Ir_setDataFlow = R"doc()doc";

static const char *__doc_popart_Ir_setDataFlow_2 = R"doc()doc";

static const char *__doc_popart_Ir_setDeviceInfo = R"doc()doc";

static const char *__doc_popart_Ir_setDeviceInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionMode = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionMode_2 = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionPhasesReady = R"doc()doc";

static const char *__doc_popart_Ir_setExecutionPhasesReady_2 = R"doc()doc";

static const char *__doc_popart_Ir_setExternalTensorDataInfo = R"doc()doc";

static const char *__doc_popart_Ir_setExternalTensorDataInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_setFinalLoss = R"doc()doc";

static const char *__doc_popart_Ir_setFinalLoss_2 = R"doc()doc";

static const char *__doc_popart_Ir_setInputShapeInfo = R"doc()doc";

static const char *__doc_popart_Ir_setInputShapeInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_setIrBundleHash = R"doc()doc";

static const char *__doc_popart_Ir_setIrBundleHash_2 = R"doc()doc";

static const char *__doc_popart_Ir_setIsPrepared = R"doc()doc";

static const char *__doc_popart_Ir_setIsPrepared_2 = R"doc()doc";

static const char *__doc_popart_Ir_setMainGraphPathFromLoss = R"doc()doc";

static const char *__doc_popart_Ir_setMainGraphPathFromLoss_2 = R"doc()doc";

static const char *__doc_popart_Ir_setNEdgesToLoss = R"doc()doc";

static const char *__doc_popart_Ir_setNEdgesToLoss_2 = R"doc()doc";

static const char *__doc_popart_Ir_setOnnxModel = R"doc()doc";

static const char *__doc_popart_Ir_setOnnxModel_2 = R"doc()doc";

static const char *__doc_popart_Ir_setOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_setOptimizer_2 = R"doc()doc";

static const char *__doc_popart_Ir_setPatterns = R"doc()doc";

static const char *__doc_popart_Ir_setPatterns_2 = R"doc()doc";

static const char *__doc_popart_Ir_setRemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_Ir_setRemoteBufferInfo_2 = R"doc()doc";

static const char *__doc_popart_Ir_setRequiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Ir_setRequiresRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Ir_setUserOptions = R"doc()doc";

static const char *__doc_popart_Ir_setUserOptions_2 = R"doc()doc";

static const char *__doc_popart_Ir_step = R"doc()doc";

static const char *__doc_popart_Ir_step_2 = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor_3 = R"doc()doc";

static const char *__doc_popart_Ir_storingIsDisabledForTensor_4 = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor_2 = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor_3 = R"doc()doc";

static const char *__doc_popart_Ir_streamingIsDisabledForTensor_4 = R"doc()doc";

static const char *__doc_popart_Ir_subgraph_id_counter = R"doc()doc";

static const char *__doc_popart_Ir_subgraph_id_counter_2 = R"doc()doc";

static const char *__doc_popart_Ir_syntheticDataMode = R"doc()doc";

static const char *__doc_popart_Ir_syntheticDataMode_2 = R"doc()doc";

static const char *__doc_popart_Ir_tensorExistsInInitialisers = R"doc()doc";

static const char *__doc_popart_Ir_tensorExistsInInitialisers_2 = R"doc()doc";

static const char *__doc_popart_Ir_timePartitionLogger =
    R"doc(Returns:
    An object used to track and summarize where wall clock time is
    spent in PopART compilation. This object is used to partition time
    into different components (scheduling, outlining, poplar Graph
    construction, etc.). It can be used as follows:

<code> void foo() { auto timer =
timePartitionLogger().scopedStopwatch("In foo"); if (cond0()){ return;
} bar(); return; } </code>

When the method timePartitionLoggerStr() (see below) is called, there
will be a line with "In foo" summarizing the time between between the
construction and destruction of *timer*, above. Something like:

In foo : 0.03 [s] : 30 % In bar : 0.02 [s] : 10 % unaccounted : 0.05
[s] : 50 % total : 0.10 [s] : 100 %.

In the case where there are multiple timers which exist concurrently,
only the most recently constructed one will accumulate time. This
means that the most nested scope is the one which will accumulate
time.

For more information, see the poprithms SwitchingTimePartitionLogger
class)doc";

static const char *__doc_popart_Ir_timePartitionLogger_2 = R"doc()doc";

static const char *__doc_popart_Ir_timePartitionLogger_3 =
    R"doc(Returns:
    An object used to track and summarize where wall clock time is
    spent in PopART compilation. This object is used to partition time
    into different components (scheduling, outlining, poplar Graph
    construction, etc.). It can be used as follows:

<code> void foo() { auto timer =
timePartitionLogger().scopedStopwatch("In foo"); if (cond0()){ return;
} bar(); return; } </code>

When the method timePartitionLoggerStr() (see below) is called, there
will be a line with "In foo" summarizing the time between between the
construction and destruction of *timer*, above. Something like:

In foo : 0.03 [s] : 30 % In bar : 0.02 [s] : 10 % unaccounted : 0.05
[s] : 50 % total : 0.10 [s] : 100 %.

In the case where there are multiple timers which exist concurrently,
only the most recently constructed one will accumulate time. This
means that the most nested scope is the one which will accumulate
time.

For more information, see the poprithms SwitchingTimePartitionLogger
class)doc";

static const char *__doc_popart_Ir_timePartitionLogger_4 = R"doc()doc";

static const char *__doc_popart_Ir_timePartitionLoggerStr = R"doc()doc";

static const char *__doc_popart_Ir_timePartitionLoggerStr_2 = R"doc()doc";

static const char *__doc_popart_Ir_transformEnableMap = R"doc()doc";

static const char *__doc_popart_Ir_transformEnableMap_2 = R"doc()doc";

static const char *__doc_popart_Ir_unsetAllVirtualGraphIds = R"doc()doc";

static const char *__doc_popart_Ir_unsetAllVirtualGraphIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_updateAliases = R"doc()doc";

static const char *__doc_popart_Ir_updateAliases_2 = R"doc()doc";

static const char *__doc_popart_Ir_updateOptimizer = R"doc()doc";

static const char *__doc_popart_Ir_updateOptimizer_2 = R"doc()doc";

static const char *__doc_popart_Ir_updateVertices = R"doc()doc";

static const char *__doc_popart_Ir_updateVertices_2 = R"doc()doc";

static const char *__doc_popart_Ir_useSyntheticData = R"doc()doc";

static const char *__doc_popart_Ir_useSyntheticData_2 = R"doc()doc";

static const char *__doc_popart_Ir_userOptions = R"doc()doc";

static const char *__doc_popart_Ir_userOptions_2 = R"doc()doc";

static const char *__doc_popart_Ir_usingEngineCache = R"doc()doc";

static const char *__doc_popart_Ir_usingEngineCache_2 = R"doc()doc";

static const char *__doc_popart_Ir_validateAnchors = R"doc()doc";

static const char *__doc_popart_Ir_validateAnchors_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyAliasZeroCopySettings = R"doc()doc";

static const char *__doc_popart_Ir_verifyAliasZeroCopySettings_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyBatchSerializationSettings =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyBatchSerializationSettings_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyConnectivity_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyConstExprFolding = R"doc()doc";

static const char *__doc_popart_Ir_verifyConstExprFolding_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyDistributedReplicatedGraphSettings =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyDistributedReplicatedGraphSettings_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyExecutionPhaseSettings = R"doc()doc";

static const char *__doc_popart_Ir_verifyExecutionPhaseSettings_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpInputConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpInputConnectivity_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpOutputConnectivity = R"doc()doc";

static const char *__doc_popart_Ir_verifyOpOutputConnectivity_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyPipelineSettings = R"doc()doc";

static const char *__doc_popart_Ir_verifyPipelineSettings_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyRecomputeAttributes = R"doc()doc";

static const char *__doc_popart_Ir_verifyRecomputeAttributes_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifySubgraphs = R"doc()doc";

static const char *__doc_popart_Ir_verifySubgraphs_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorConsumerConnectivity =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorConsumerConnectivity_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorIds = R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorProducerConnectivity =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyTensorProducerConnectivity_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVertexAttributesOnlyInMain =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVertexAttributesOnlyInMain_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVirtualGraphIds = R"doc()doc";

static const char *__doc_popart_Ir_verifyVirtualGraphIds_2 = R"doc()doc";

static const char *__doc_popart_Ir_verifyVirualGraphIdsNotInitialized =
    R"doc()doc";

static const char *__doc_popart_Ir_verifyVirualGraphIdsNotInitialized_2 =
    R"doc()doc";

static const char *__doc_popart_Ir_virtualGraphsEnabled = R"doc()doc";

static const char *__doc_popart_Ir_virtualGraphsEnabled_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp = R"doc()doc";

static const char *__doc_popart_L1GradOp_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_L1GradOp = R"doc()doc";

static const char *__doc_popart_L1GradOp_L1GradOp_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_canShard = R"doc()doc";

static const char *__doc_popart_L1GradOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_clone = R"doc()doc";

static const char *__doc_popart_L1GradOp_clone_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getFwdActInIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getFwdActInIndex_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getGradInIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getGradInIndex_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getLambda = R"doc()doc";

static const char *__doc_popart_L1GradOp_getLambda_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_L1GradOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getReductionType = R"doc()doc";

static const char *__doc_popart_L1GradOp_getReductionType_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getScaleByReplication = R"doc()doc";

static const char *__doc_popart_L1GradOp_getScaleByReplication_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getShardRescaleFactor = R"doc()doc";

static const char *__doc_popart_L1GradOp_getShardRescaleFactor_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_L1GradOp_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradInputInfo_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_L1GradOp_gradOutToNonGradIn_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_lambda = R"doc()doc";

static const char *__doc_popart_L1GradOp_lambda_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_reduction = R"doc()doc";

static const char *__doc_popart_L1GradOp_reduction_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_scaleByReplication = R"doc()doc";

static const char *__doc_popart_L1GradOp_scaleByReplication_2 = R"doc()doc";

static const char *__doc_popart_L1GradOp_setup = R"doc()doc";

static const char *__doc_popart_L1GradOp_setup_2 = R"doc()doc";

static const char *__doc_popart_L1Op = R"doc()doc";

static const char *__doc_popart_L1Op_2 = R"doc()doc";

static const char *__doc_popart_L1Op_L1Op = R"doc()doc";

static const char *__doc_popart_L1Op_L1Op_2 = R"doc()doc";

static const char *__doc_popart_L1Op_canShard = R"doc()doc";

static const char *__doc_popart_L1Op_canShard_2 = R"doc()doc";

static const char *__doc_popart_L1Op_clone = R"doc()doc";

static const char *__doc_popart_L1Op_clone_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getGradOps = R"doc()doc";

static const char *__doc_popart_L1Op_getGradOps_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getInIndex = R"doc()doc";

static const char *__doc_popart_L1Op_getInIndex_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getLambda = R"doc()doc";

static const char *__doc_popart_L1Op_getLambda_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getOutIndex = R"doc()doc";

static const char *__doc_popart_L1Op_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getShardReductionType = R"doc()doc";

static const char *__doc_popart_L1Op_getShardReductionType_2 = R"doc()doc";

static const char *__doc_popart_L1Op_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_L1Op_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_L1Op_lambda = R"doc()doc";

static const char *__doc_popart_L1Op_lambda_2 = R"doc()doc";

static const char *__doc_popart_L1Op_setup = R"doc()doc";

static const char *__doc_popart_L1Op_setup_2 = R"doc()doc";

static const char *__doc_popart_MergeVarUpdateType =
    R"doc(Enum type used to specify which `VarUpdateOp` ops to merge.)doc";

static const char *__doc_popart_MergeVarUpdateType_2 =
    R"doc(Enum type used to specify which `VarUpdateOp` ops to merge.)doc";

static const char *__doc_popart_MergeVarUpdateType_All =
    R"doc(Merge all VarUpdateOp ops into as few groups as possible. This is a
good choice when memory is not a constraint.)doc";

static const char *__doc_popart_MergeVarUpdateType_All_2 =
    R"doc(Merge all VarUpdateOp ops into as few groups as possible. This is a
good choice when memory is not a constraint.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoLoose =
    R"doc(Merge into groups while attempting not to increase maximum variable
liveness, and also not slice tensor variables so they they will need
to be processed by different VarUpdateOp ops.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoLoose_2 =
    R"doc(Merge into groups while attempting not to increase maximum variable
liveness, and also not slice tensor variables so they they will need
to be processed by different VarUpdateOp ops.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoTight =
    R"doc(Merge into groups, so that VarUpdateOp ops process tensors of exactly
`mergeVarUpdateMemThreshold` in size.)doc";

static const char *__doc_popart_MergeVarUpdateType_AutoTight_2 =
    R"doc(Merge into groups, so that VarUpdateOp ops process tensors of exactly
`mergeVarUpdateMemThreshold` in size.)doc";

static const char *__doc_popart_MergeVarUpdateType_N =
    R"doc(The number of ``MergeVarUpdateTypes`` values.)doc";

static const char *__doc_popart_MergeVarUpdateType_N_2 =
    R"doc(The number of ``MergeVarUpdateTypes`` values.)doc";

static const char *__doc_popart_MergeVarUpdateType_None =
    R"doc(Do not merge VarUpdateOp ops.)doc";

static const char *__doc_popart_MergeVarUpdateType_None_2 =
    R"doc(Do not merge VarUpdateOp ops.)doc";

static const char *__doc_popart_NllGradOp = R"doc()doc";

static const char *__doc_popart_NllGradOp_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_NllGradOp = R"doc()doc";

static const char *__doc_popart_NllGradOp_NllGradOp_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_NllGradOp_appendOutlineAttributes_2 =
    R"doc()doc";

static const char *__doc_popart_NllGradOp_canShard = R"doc()doc";

static const char *__doc_popart_NllGradOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_clone = R"doc()doc";

static const char *__doc_popart_NllGradOp_clone_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getGradInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getGradInIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getIgnoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLabelInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLabelInIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLossTensorId = R"doc()doc";

static const char *__doc_popart_NllGradOp_getLossTensorId_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getOptionalIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getOptionalIgnoreIndex_2 =
    R"doc()doc";

static const char *__doc_popart_NllGradOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getProbsInIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_getProbsInIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getReductionType = R"doc()doc";

static const char *__doc_popart_NllGradOp_getReductionType_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getScaleByReplication = R"doc()doc";

static const char *__doc_popart_NllGradOp_getScaleByReplication_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getShardRescaleFactor = R"doc()doc";

static const char *__doc_popart_NllGradOp_getShardRescaleFactor_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_NllGradOp_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradInputInfo = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradInputInfo_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_NllGradOp_gradOutToNonGradIn_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_hasIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_hasIgnoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_ignoreIndex = R"doc()doc";

static const char *__doc_popart_NllGradOp_ignoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability_3 = R"doc()doc";

static const char *__doc_popart_NllGradOp_inputIsLogProbability_4 = R"doc()doc";

static const char *__doc_popart_NllGradOp_lossId = R"doc()doc";

static const char *__doc_popart_NllGradOp_lossId_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_reduction = R"doc()doc";

static const char *__doc_popart_NllGradOp_reduction_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_scaleByReplication = R"doc()doc";

static const char *__doc_popart_NllGradOp_scaleByReplication_2 = R"doc()doc";

static const char *__doc_popart_NllGradOp_setup = R"doc()doc";

static const char *__doc_popart_NllGradOp_setup_2 = R"doc()doc";

static const char *__doc_popart_NllOp = R"doc()doc";

static const char *__doc_popart_NllOp_2 = R"doc()doc";

static const char *__doc_popart_NllOp_NllOp = R"doc()doc";

static const char *__doc_popart_NllOp_NllOp_2 = R"doc()doc";

static const char *__doc_popart_NllOp_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_NllOp_appendOutlineAttributes_2 = R"doc()doc";

static const char *__doc_popart_NllOp_canShard = R"doc()doc";

static const char *__doc_popart_NllOp_canShard_2 = R"doc()doc";

static const char *__doc_popart_NllOp_clone = R"doc()doc";

static const char *__doc_popart_NllOp_clone_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getGradOps = R"doc()doc";

static const char *__doc_popart_NllOp_getGradOps_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getIgnoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getLabelInIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getLabelInIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getOptionalIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getOptionalIgnoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getOutIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getOutIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getProbsInIndex = R"doc()doc";

static const char *__doc_popart_NllOp_getProbsInIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getShardReductionType = R"doc()doc";

static const char *__doc_popart_NllOp_getShardReductionType_2 = R"doc()doc";

static const char *__doc_popart_NllOp_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_NllOp_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_NllOp_hasIgnoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_hasIgnoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_ignoreIndex = R"doc()doc";

static const char *__doc_popart_NllOp_ignoreIndex_2 = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability_2 = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability_3 = R"doc()doc";

static const char *__doc_popart_NllOp_inputIsLogProbability_4 = R"doc()doc";

static const char *__doc_popart_NllOp_setup = R"doc()doc";

static const char *__doc_popart_NllOp_setup_2 = R"doc()doc";

static const char *__doc_popart_Op = R"doc()doc";

static const char *__doc_popart_Op_2 = R"doc()doc";

static const char *__doc_popart_OpCreator = R"doc()doc";

static const char *__doc_popart_OpCreator_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_OpCreatorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_OpCreatorInfo_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_attributes = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_attributes_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputData = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputData_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue_2 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue_3 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputScalarValue_4 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensor = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensor_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorData = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorData_2 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorInfo = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getInputTensorInfo_2 =
    R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getOutputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_getOutputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_hasInputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_hasInputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_hasOutputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_hasOutputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_inputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_inputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_opid = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_opid_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_outputIds = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_outputIds_2 = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_settings = R"doc()doc";

static const char *__doc_popart_OpCreatorInfo_settings_2 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_2 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_3 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_4 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_5 = R"doc()doc";

static const char *__doc_popart_OpCreator_OpCreator_6 = R"doc()doc";

static const char *__doc_popart_OpDefinition = R"doc()doc";

static const char *__doc_popart_OpDefinition_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_Attribute = R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_Attribute_2 =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_supportedValuesRegex =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Attribute_supportedValuesRegex_2 =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Input = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_Input = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_Input_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_constant = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_constant_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_name = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_name_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_supportedTensors =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Input_supportedTensors_2 =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition = R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition_3 = R"doc()doc";

static const char *__doc_popart_OpDefinition_OpDefinition_4 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_Output = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_Output_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_name = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_name_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_supportedTensors =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_Output_supportedTensors_2 =
    R"doc()doc";

static const char *__doc_popart_OpDefinition_attributes = R"doc()doc";

static const char *__doc_popart_OpDefinition_attributes_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_inputs = R"doc()doc";

static const char *__doc_popart_OpDefinition_inputs_2 = R"doc()doc";

static const char *__doc_popart_OpDefinition_outputs = R"doc()doc";

static const char *__doc_popart_OpDefinition_outputs_2 = R"doc()doc";

static const char *__doc_popart_OpManager = R"doc()doc";

static const char *__doc_popart_OpManager_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo_3 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_OpInfo_4 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_complexFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_complexFactory_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_details = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_details_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getComplexFactory =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getComplexFactory_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getSimpleFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_getSimpleFactory_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_hasComplexFactory =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_hasComplexFactory_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_id = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_id_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_isPublic = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_isPublic_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_simpleFactory = R"doc()doc";

static const char *__doc_popart_OpManager_OpInfo_simpleFactory_2 = R"doc()doc";

static const char *__doc_popart_OpManager_OpManager = R"doc()doc";

static const char *__doc_popart_OpManager_OpManager_2 = R"doc()doc";

static const char *__doc_popart_OpManager_checkOpVersionAgainstOpset =
    R"doc()doc";

static const char *__doc_popart_OpManager_checkOpVersionAgainstOpset_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_create = R"doc()doc";

static const char *__doc_popart_OpManager_create_2 = R"doc()doc";

static const char *__doc_popart_OpManager_create_3 = R"doc()doc";

static const char *__doc_popart_OpManager_create_4 = R"doc()doc";

static const char *__doc_popart_OpManager_createOp = R"doc()doc";

static const char *__doc_popart_OpManager_createOp_2 = R"doc()doc";

static const char *__doc_popart_OpManager_createOp_3 = R"doc()doc";

static const char *__doc_popart_OpManager_createOp_4 = R"doc()doc";

static const char *__doc_popart_OpManager_createOpInGraph = R"doc()doc";

static const char *__doc_popart_OpManager_createOpInGraph_2 = R"doc()doc";

static const char *__doc_popart_OpManager_createOpWithInputs = R"doc()doc";

static const char *__doc_popart_OpManager_createOpWithInputs_2 = R"doc()doc";

static const char *__doc_popart_OpManager_findOpInfo = R"doc()doc";

static const char *__doc_popart_OpManager_findOpInfo_2 = R"doc()doc";

static const char *__doc_popart_OpManager_getAttributesFromAnyMap = R"doc()doc";

static const char *__doc_popart_OpManager_getAttributesFromAnyMap_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_getInstance = R"doc()doc";

static const char *__doc_popart_OpManager_getInstance_2 = R"doc()doc";

static const char *__doc_popart_OpManager_getOpVersionFromOpSet = R"doc()doc";

static const char *__doc_popart_OpManager_getOpVersionFromOpSet_2 = R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperations = R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperations_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperationsDefinition =
    R"doc()doc";

static const char *__doc_popart_OpManager_getSupportedOperationsDefinition_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_getUnsupportedOperations =
    R"doc()doc";

static const char *__doc_popart_OpManager_getUnsupportedOperations_2 =
    R"doc()doc";

static const char *__doc_popart_OpManager_opMap = R"doc()doc";

static const char *__doc_popart_OpManager_opMap_2 = R"doc()doc";

static const char *__doc_popart_OpManager_registerOp = R"doc()doc";

static const char *__doc_popart_OpManager_registerOp_2 = R"doc()doc";

static const char *__doc_popart_OpSerialiserBase = R"doc()doc";

static const char *__doc_popart_OpSerialiserBase_2 = R"doc()doc";

static const char *__doc_popart_Op_Op = R"doc()doc";

static const char *__doc_popart_Op_Op_2 = R"doc()doc";

static const char *__doc_popart_Op_Op_3 = R"doc()doc";

static const char *__doc_popart_Op_Op_4 = R"doc()doc";

static const char *__doc_popart_Op_Settings = R"doc()doc";

static const char *__doc_popart_Op_Settings_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings_3 = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings_4 = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings_5 = R"doc()doc";

static const char *__doc_popart_Op_Settings_Settings_6 = R"doc()doc";

static const char *__doc_popart_Op_Settings_batchSerializedPhase = R"doc()doc";

static const char *__doc_popart_Op_Settings_batchSerializedPhase_2 =
    R"doc()doc";

static const char *__doc_popart_Op_Settings_debugInfoId = R"doc()doc";

static const char *__doc_popart_Op_Settings_debugInfoId_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_excludePatterns = R"doc()doc";

static const char *__doc_popart_Op_Settings_excludePatterns_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_executionContext = R"doc()doc";

static const char *__doc_popart_Op_Settings_executionContext_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_executionPhase = R"doc()doc";

static const char *__doc_popart_Op_Settings_executionPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_extraOutlineAttributes =
    R"doc()doc";

static const char *__doc_popart_Op_Settings_extraOutlineAttributes_2 =
    R"doc()doc";

static const char *__doc_popart_Op_Settings_getIr = R"doc()doc";

static const char *__doc_popart_Op_Settings_getIr_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_graph = R"doc()doc";

static const char *__doc_popart_Op_Settings_graph_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_inferTensorMappingToFrom =
    R"doc()doc";

static const char *__doc_popart_Op_Settings_inferTensorMappingToFrom_2 =
    R"doc()doc";

static const char *__doc_popart_Op_Settings_inplacePriorityVeto = R"doc()doc";

static const char *__doc_popart_Op_Settings_inplacePriorityVeto_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_name = R"doc()doc";

static const char *__doc_popart_Op_Settings_name_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_optimizerOp = R"doc()doc";

static const char *__doc_popart_Op_Settings_optimizerOp_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_pipelineStage = R"doc()doc";

static const char *__doc_popart_Op_Settings_pipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_recomputeType = R"doc()doc";

static const char *__doc_popart_Op_Settings_recomputeType_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_schedulePriority = R"doc()doc";

static const char *__doc_popart_Op_Settings_schedulePriority_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_scope = R"doc()doc";

static const char *__doc_popart_Op_Settings_scope_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_setFromAttributes = R"doc()doc";

static const char *__doc_popart_Op_Settings_setFromAttributes_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_tensorLocation = R"doc()doc";

static const char *__doc_popart_Op_Settings_tensorLocation_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_tileSet = R"doc()doc";

static const char *__doc_popart_Op_Settings_tileSet_2 = R"doc()doc";

static const char *__doc_popart_Op_Settings_vgraphId = R"doc()doc";

static const char *__doc_popart_Op_Settings_vgraphId_2 = R"doc()doc";

static const char *__doc_popart_Op_adjustInSettings = R"doc()doc";

static const char *__doc_popart_Op_adjustInSettings_2 = R"doc()doc";

static const char *__doc_popart_Op_adjustOutSettings = R"doc()doc";

static const char *__doc_popart_Op_adjustOutSettings_2 = R"doc()doc";

static const char *__doc_popart_Op_adjustShardPlans = R"doc()doc";

static const char *__doc_popart_Op_adjustShardPlans_2 = R"doc()doc";

static const char *__doc_popart_Op_aliases = R"doc()doc";

static const char *__doc_popart_Op_aliases_2 = R"doc()doc";

static const char *__doc_popart_Op_append = R"doc()doc";

static const char *__doc_popart_Op_append_2 = R"doc()doc";

static const char *__doc_popart_Op_appendAttributes = R"doc()doc";

static const char *__doc_popart_Op_appendAttributes_2 = R"doc()doc";

static const char *__doc_popart_Op_appendMore = R"doc()doc";

static const char *__doc_popart_Op_appendMore_2 = R"doc()doc";

static const char *__doc_popart_Op_appendOutlineAttributes = R"doc()doc";

static const char *__doc_popart_Op_appendOutlineAttributes_2 = R"doc()doc";

static const char *__doc_popart_Op_bwdRegMap = R"doc()doc";

static const char *__doc_popart_Op_bwdRegMap_2 = R"doc()doc";

static const char *__doc_popart_Op_canBeReplacedByIdentity = R"doc()doc";

static const char *__doc_popart_Op_canBeReplacedByIdentity_2 = R"doc()doc";

static const char *__doc_popart_Op_canShard = R"doc()doc";

static const char *__doc_popart_Op_canShard_2 = R"doc()doc";

static const char *__doc_popart_Op_clone = R"doc()doc";

static const char *__doc_popart_Op_clone_2 = R"doc()doc";

static const char *__doc_popart_Op_configureForReplicatedTensorSharding =
    R"doc()doc";

static const char *__doc_popart_Op_configureForReplicatedTensorSharding_2 =
    R"doc()doc";

static const char *__doc_popart_Op_configureShardedOp = R"doc()doc";

static const char *__doc_popart_Op_configureShardedOp_2 = R"doc()doc";

static const char *__doc_popart_Op_connectInTensor = R"doc()doc";

static const char *__doc_popart_Op_connectInTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_connectOutTensor = R"doc()doc";

static const char *__doc_popart_Op_connectOutTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_consumesGraphOutput = R"doc()doc";

static const char *__doc_popart_Op_consumesGraphOutput_2 = R"doc()doc";

static const char *__doc_popart_Op_copiesOptimizerTensors = R"doc()doc";

static const char *__doc_popart_Op_copiesOptimizerTensors_2 = R"doc()doc";

static const char *__doc_popart_Op_createAndConnectOutTensor = R"doc()doc";

static const char *__doc_popart_Op_createAndConnectOutTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_debugInfo = R"doc()doc";

static const char *__doc_popart_Op_debugInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_debugName = R"doc()doc";

static const char *__doc_popart_Op_debugName_2 = R"doc()doc";

static const char *__doc_popart_Op_defaultConnectInTensor = R"doc()doc";

static const char *__doc_popart_Op_defaultConnectInTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_disconnectAllInputs = R"doc()doc";

static const char *__doc_popart_Op_disconnectAllInputs_2 = R"doc()doc";

static const char *__doc_popart_Op_disconnectAllOutputs = R"doc()doc";

static const char *__doc_popart_Op_disconnectAllOutputs_2 = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor_3 = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor_4 = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor_5 = R"doc()doc";

static const char *__doc_popart_Op_disconnectInTensor_6 = R"doc()doc";

static const char *__doc_popart_Op_disconnectOutTensor = R"doc()doc";

static const char *__doc_popart_Op_disconnectOutTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_doesAlias =
    R"doc(Returns:
    True if there is an input which aliases an output.)doc";

static const char *__doc_popart_Op_doesAlias_2 =
    R"doc(Returns:
    True if the input at \p inIndex aliases the output at \p outIndex.)doc";

static const char *__doc_popart_Op_doesAlias_3 =
    R"doc(Returns:
    True if there is an input which aliases an output.)doc";

static const char *__doc_popart_Op_doesAlias_4 =
    R"doc(Returns:
    True if the input at \p inIndex aliases the output at \p outIndex.)doc";

static const char *__doc_popart_Op_finalizeDebugInfo = R"doc()doc";

static const char *__doc_popart_Op_finalizeDebugInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_fwdRegMap = R"doc()doc";

static const char *__doc_popart_Op_fwdRegMap_2 = R"doc()doc";

static const char *__doc_popart_Op_getBatchSerializedPhase = R"doc()doc";

static const char *__doc_popart_Op_getBatchSerializedPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_getCalledGraphs = R"doc()doc";

static const char *__doc_popart_Op_getCalledGraphs_2 = R"doc()doc";

static const char *__doc_popart_Op_getDebugInfo = R"doc()doc";

static const char *__doc_popart_Op_getDebugInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_getExecutionPhase = R"doc()doc";

static const char *__doc_popart_Op_getExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_getGradOps = R"doc()doc";

static const char *__doc_popart_Op_getGradOps_2 = R"doc()doc";

static const char *__doc_popart_Op_getGraph = R"doc()doc";

static const char *__doc_popart_Op_getGraph_2 = R"doc()doc";

static const char *__doc_popart_Op_getGraph_3 = R"doc()doc";

static const char *__doc_popart_Op_getGraph_4 = R"doc()doc";

static const char *__doc_popart_Op_getHighSubgraphValue = R"doc()doc";

static const char *__doc_popart_Op_getHighSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_Op_getInBatchAxis = R"doc()doc";

static const char *__doc_popart_Op_getInBatchAxis_2 = R"doc()doc";

static const char *__doc_popart_Op_getInSettings = R"doc()doc";

static const char *__doc_popart_Op_getInSettings_2 = R"doc()doc";

static const char *__doc_popart_Op_getInTensorData = R"doc()doc";

static const char *__doc_popart_Op_getInTensorData_2 = R"doc()doc";

static const char *__doc_popart_Op_getInplaceVariant = R"doc()doc";

static const char *__doc_popart_Op_getInplaceVariant_2 = R"doc()doc";

static const char *__doc_popart_Op_getIntrospectionInVirtualGraphId =
    R"doc()doc";

static const char *__doc_popart_Op_getIntrospectionInVirtualGraphId_2 =
    R"doc()doc";

static const char *__doc_popart_Op_getIntrospectionOutVirtualGraphId =
    R"doc()doc";

static const char *__doc_popart_Op_getIntrospectionOutVirtualGraphId_2 =
    R"doc()doc";

static const char *__doc_popart_Op_getIr = R"doc()doc";

static const char *__doc_popart_Op_getIr_2 = R"doc()doc";

static const char *__doc_popart_Op_getIr_3 = R"doc()doc";

static const char *__doc_popart_Op_getIr_4 = R"doc()doc";

static const char *__doc_popart_Op_getLowSubgraphValue = R"doc()doc";

static const char *__doc_popart_Op_getLowSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_Op_getName = R"doc()doc";

static const char *__doc_popart_Op_getName_2 = R"doc()doc";

static const char *__doc_popart_Op_getNonGradInIndex = R"doc()doc";

static const char *__doc_popart_Op_getNonGradInIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_getOptionalBatchSerializedPhase =
    R"doc()doc";

static const char *__doc_popart_Op_getOptionalBatchSerializedPhase_2 =
    R"doc()doc";

static const char *__doc_popart_Op_getOptionalExecutionPhase = R"doc()doc";

static const char *__doc_popart_Op_getOptionalExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_getOptionalPipelineStage = R"doc()doc";

static const char *__doc_popart_Op_getOptionalPipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Op_getOptionalVGraphId = R"doc()doc";

static const char *__doc_popart_Op_getOptionalVGraphId_2 = R"doc()doc";

static const char *__doc_popart_Op_getOutBatchAxis = R"doc()doc";

static const char *__doc_popart_Op_getOutBatchAxis_2 = R"doc()doc";

static const char *__doc_popart_Op_getOutSettings = R"doc()doc";

static const char *__doc_popart_Op_getOutSettings_2 = R"doc()doc";

static const char *__doc_popart_Op_getPipelineStage = R"doc()doc";

static const char *__doc_popart_Op_getPipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Op_getReplicatedTensorShardingIndices =
    R"doc()doc";

static const char *__doc_popart_Op_getReplicatedTensorShardingIndices_2 =
    R"doc()doc";

static const char *__doc_popart_Op_getScope = R"doc()doc";

static const char *__doc_popart_Op_getScope_2 = R"doc()doc";

static const char *__doc_popart_Op_getSeedInIndex = R"doc()doc";

static const char *__doc_popart_Op_getSeedInIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_getSettings = R"doc()doc";

static const char *__doc_popart_Op_getSettings_2 = R"doc()doc";

static const char *__doc_popart_Op_getSettings_3 = R"doc()doc";

static const char *__doc_popart_Op_getSettings_4 = R"doc()doc";

static const char *__doc_popart_Op_getShardReductionType = R"doc()doc";

static const char *__doc_popart_Op_getShardReductionType_2 = R"doc()doc";

static const char *__doc_popart_Op_getShardRescaleFactor = R"doc()doc";

static const char *__doc_popart_Op_getShardRescaleFactor_2 = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphEquivId = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphEquivId_2 = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphInputs = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphInputs_2 = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphOutputs = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphOutputs_2 = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphValue = R"doc()doc";

static const char *__doc_popart_Op_getSubgraphValue_2 = R"doc()doc";

static const char *__doc_popart_Op_getVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Op_getVirtualGraphId_2 = R"doc()doc";

static const char *__doc_popart_Op_gradInputInfo = R"doc()doc";

static const char *__doc_popart_Op_gradInputInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_gradOutToNonGradIn = R"doc()doc";

static const char *__doc_popart_Op_gradOutToNonGradIn_2 = R"doc()doc";

static const char *__doc_popart_Op_hasAliasedModifiers =
    R"doc(Check if output is modified by any consumer.

Parameter ``out``:
    OutIndex to check.

Returns:
    True if any consumer of any aliased tensor downstream modifies a
    non-empty region, false otherwise.)doc";

static const char *__doc_popart_Op_hasAliasedModifiers_2 =
    R"doc(Check if output is modified by any consumer.

Parameter ``out``:
    OutIndex to check.

Returns:
    True if any consumer of any aliased tensor downstream modifies a
    non-empty region, false otherwise.)doc";

static const char *__doc_popart_Op_hasBatchSerializedPhase = R"doc()doc";

static const char *__doc_popart_Op_hasBatchSerializedPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_hasExecutionPhase = R"doc()doc";

static const char *__doc_popart_Op_hasExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_hasInput = R"doc()doc";

static const char *__doc_popart_Op_hasInput_2 = R"doc()doc";

static const char *__doc_popart_Op_hasOutput = R"doc()doc";

static const char *__doc_popart_Op_hasOutput_2 = R"doc()doc";

static const char *__doc_popart_Op_hasPipelineStage = R"doc()doc";

static const char *__doc_popart_Op_hasPipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Op_hasSideEffect = R"doc()doc";

static const char *__doc_popart_Op_hasSideEffect_2 = R"doc()doc";

static const char *__doc_popart_Op_hasVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Op_hasVirtualGraphId_2 = R"doc()doc";

static const char *__doc_popart_Op_id = R"doc()doc";

static const char *__doc_popart_Op_id_2 = R"doc()doc";

static const char *__doc_popart_Op_inId = R"doc()doc";

static const char *__doc_popart_Op_inId_2 = R"doc()doc";

static const char *__doc_popart_Op_inId_3 = R"doc()doc";

static const char *__doc_popart_Op_inId_4 = R"doc()doc";

static const char *__doc_popart_Op_inInfo = R"doc()doc";

static const char *__doc_popart_Op_inInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_inInfo_3 = R"doc()doc";

static const char *__doc_popart_Op_inInfo_4 = R"doc()doc";

static const char *__doc_popart_Op_inRank = R"doc()doc";

static const char *__doc_popart_Op_inRank_2 = R"doc()doc";

static const char *__doc_popart_Op_inShape = R"doc()doc";

static const char *__doc_popart_Op_inShape_2 = R"doc()doc";

static const char *__doc_popart_Op_inTensor = R"doc()doc";

static const char *__doc_popart_Op_inTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_inTensor_3 = R"doc()doc";

static const char *__doc_popart_Op_inTensor_4 = R"doc()doc";

static const char *__doc_popart_Op_inTensorCount = R"doc()doc";

static const char *__doc_popart_Op_inTensorCount_2 = R"doc()doc";

static const char *__doc_popart_Op_inheritPlacementAttributes = R"doc()doc";

static const char *__doc_popart_Op_inheritPlacementAttributes_2 = R"doc()doc";

static const char *__doc_popart_Op_inplacePriorityDefault = R"doc()doc";

static const char *__doc_popart_Op_inplacePriorityDefault_2 = R"doc()doc";

static const char *__doc_popart_Op_input = R"doc()doc";

static const char *__doc_popart_Op_input_2 = R"doc()doc";

static const char *__doc_popart_Op_inputUnmodifiable =
    R"doc(Check if input is unmodifiable or aliases an unmodifiable tensor.

Parameter ``in``:
    InIndex to check.

Returns:
    True if any connected variable tensor has a non-empty alias chain
    and is unmodifiable, false otherwise.)doc";

static const char *__doc_popart_Op_inputUnmodifiable_2 =
    R"doc(Check if input is unmodifiable or aliases an unmodifiable tensor.

Parameter ``in``:
    InIndex to check.

Returns:
    True if any connected variable tensor has a non-empty alias chain
    and is unmodifiable, false otherwise.)doc";

static const char *__doc_popart_Op_inputsUnmodifiable = R"doc()doc";

static const char *__doc_popart_Op_inputsUnmodifiable_2 = R"doc()doc";

static const char *__doc_popart_Op_isChildOf = R"doc()doc";

static const char *__doc_popart_Op_isChildOf_2 = R"doc()doc";

static const char *__doc_popart_Op_isConvertibleTo = R"doc()doc";

static const char *__doc_popart_Op_isConvertibleTo_2 = R"doc()doc";

static const char *__doc_popart_Op_isElementWiseUnary = R"doc()doc";

static const char *__doc_popart_Op_isElementWiseUnary_2 = R"doc()doc";

static const char *__doc_popart_Op_isExcludedFromPattern = R"doc()doc";

static const char *__doc_popart_Op_isExcludedFromPattern_2 = R"doc()doc";

static const char *__doc_popart_Op_isIpuCopyOp = R"doc()doc";

static const char *__doc_popart_Op_isIpuCopyOp_2 = R"doc()doc";

static const char *__doc_popart_Op_isLossOp = R"doc()doc";

static const char *__doc_popart_Op_isLossOp_2 = R"doc()doc";

static const char *__doc_popart_Op_isNorm = R"doc()doc";

static const char *__doc_popart_Op_isNorm_2 = R"doc()doc";

static const char *__doc_popart_Op_isOptimizerOp = R"doc()doc";

static const char *__doc_popart_Op_isOptimizerOp_2 = R"doc()doc";

static const char *__doc_popart_Op_isOutlineable = R"doc()doc";

static const char *__doc_popart_Op_isOutlineable_2 = R"doc()doc";

static const char *__doc_popart_Op_isOutplace = R"doc()doc";

static const char *__doc_popart_Op_isOutplace_2 = R"doc()doc";

static const char *__doc_popart_Op_isParentOf = R"doc()doc";

static const char *__doc_popart_Op_isParentOf_2 = R"doc()doc";

static const char *__doc_popart_Op_loopShard = R"doc()doc";

static const char *__doc_popart_Op_loopShard_2 = R"doc()doc";

static const char *__doc_popart_Op_memOfOutputs = R"doc()doc";

static const char *__doc_popart_Op_memOfOutputs_2 = R"doc()doc";

static const char *__doc_popart_Op_modifies = R"doc()doc";

static const char *__doc_popart_Op_modifies_2 =
    R"doc(Is modifies(i) non-empty for any input index i?

Returns:
    True if modifies(i) is non-empty for any i, false otherwise.)doc";

static const char *__doc_popart_Op_modifies_3 = R"doc()doc";

static const char *__doc_popart_Op_modifies_4 =
    R"doc(Is modifies(i) non-empty for any input index i?

Returns:
    True if modifies(i) is non-empty for any i, false otherwise.)doc";

static const char *__doc_popart_Op_modifiesIndex =
    R"doc(Check if an op modifies a tensor at a specific index in.

Parameter ``in``:
    Index to check.

Returns:
    True if it modifies the tensor, false otherwise.)doc";

static const char *__doc_popart_Op_modifiesIndex_2 =
    R"doc(Check if an op modifies a tensor at a specific index in.

Parameter ``in``:
    Index to check.

Returns:
    True if it modifies the tensor, false otherwise.)doc";

static const char *__doc_popart_Op_name = R"doc()doc";

static const char *__doc_popart_Op_name_2 = R"doc()doc";

static const char *__doc_popart_Op_opInToSubgraphInIndex = R"doc()doc";

static const char *__doc_popart_Op_opInToSubgraphInIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_opOutToSubgraphOutIndex = R"doc()doc";

static const char *__doc_popart_Op_opOutToSubgraphOutIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_operator_assign = R"doc()doc";

static const char *__doc_popart_Op_operator_assign_2 = R"doc()doc";

static const char *__doc_popart_Op_opid = R"doc()doc";

static const char *__doc_popart_Op_opid_2 = R"doc()doc";

static const char *__doc_popart_Op_optionalInputs = R"doc()doc";

static const char *__doc_popart_Op_optionalInputs_2 = R"doc()doc";

static const char *__doc_popart_Op_outId = R"doc()doc";

static const char *__doc_popart_Op_outId_2 = R"doc()doc";

static const char *__doc_popart_Op_outId_3 = R"doc()doc";

static const char *__doc_popart_Op_outId_4 = R"doc()doc";

static const char *__doc_popart_Op_outIndex = R"doc()doc";

static const char *__doc_popart_Op_outIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_outInfo = R"doc()doc";

static const char *__doc_popart_Op_outInfo_2 = R"doc()doc";

static const char *__doc_popart_Op_outInfo_3 = R"doc()doc";

static const char *__doc_popart_Op_outInfo_4 = R"doc()doc";

static const char *__doc_popart_Op_outRank = R"doc()doc";

static const char *__doc_popart_Op_outRank_2 = R"doc()doc";

static const char *__doc_popart_Op_outShape = R"doc()doc";

static const char *__doc_popart_Op_outShape_2 = R"doc()doc";

static const char *__doc_popart_Op_outTensor = R"doc()doc";

static const char *__doc_popart_Op_outTensor_2 = R"doc()doc";

static const char *__doc_popart_Op_outTensor_3 = R"doc()doc";

static const char *__doc_popart_Op_outTensor_4 = R"doc()doc";

static const char *__doc_popart_Op_outTensorCount = R"doc()doc";

static const char *__doc_popart_Op_outTensorCount_2 = R"doc()doc";

static const char *__doc_popart_Op_output = R"doc()doc";

static const char *__doc_popart_Op_output_2 = R"doc()doc";

static const char *__doc_popart_Op_overwritesTensor =
    R"doc(Check if an op overwrites a tensor at a specific index in.

Parameter ``t``:
    Tensor to check.

Returns:
    True if it overwrites the tensor, false otherwise.)doc";

static const char *__doc_popart_Op_overwritesTensor_2 =
    R"doc(Check if an op overwrites a tensor at a specific index in.

Parameter ``t``:
    Tensor to check.

Returns:
    True if it overwrites the tensor, false otherwise.)doc";

static const char *__doc_popart_Op_prettyNpOut = R"doc()doc";

static const char *__doc_popart_Op_prettyNpOut_2 = R"doc()doc";

static const char *__doc_popart_Op_prettyNpOut_3 = R"doc()doc";

static const char *__doc_popart_Op_prettyNpOut_4 = R"doc()doc";

static const char *__doc_popart_Op_producesGraphOutput = R"doc()doc";

static const char *__doc_popart_Op_producesGraphOutput_2 = R"doc()doc";

static const char *__doc_popart_Op_pruneable = R"doc()doc";

static const char *__doc_popart_Op_pruneable_2 = R"doc()doc";

static const char *__doc_popart_Op_readyToCreateGradients = R"doc()doc";

static const char *__doc_popart_Op_readyToCreateGradients_2 = R"doc()doc";

static const char *__doc_popart_Op_requiresRandomSeed = R"doc()doc";

static const char *__doc_popart_Op_requiresRandomSeed_2 = R"doc()doc";

static const char *__doc_popart_Op_setBatchSerializedPhase = R"doc()doc";

static const char *__doc_popart_Op_setBatchSerializedPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_setExecutionPhase = R"doc()doc";

static const char *__doc_popart_Op_setExecutionPhase_2 = R"doc()doc";

static const char *__doc_popart_Op_setName = R"doc()doc";

static const char *__doc_popart_Op_setName_2 = R"doc()doc";

static const char *__doc_popart_Op_setPipelineStage = R"doc()doc";

static const char *__doc_popart_Op_setPipelineStage_2 = R"doc()doc";

static const char *__doc_popart_Op_setScope = R"doc()doc";

static const char *__doc_popart_Op_setScope_2 = R"doc()doc";

static const char *__doc_popart_Op_setVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Op_setVirtualGraphId_2 = R"doc()doc";

static const char *__doc_popart_Op_settings = R"doc()doc";

static const char *__doc_popart_Op_settings_2 = R"doc()doc";

static const char *__doc_popart_Op_setup = R"doc()doc";

static const char *__doc_popart_Op_setup_2 = R"doc()doc";

static const char *__doc_popart_Op_shard = R"doc()doc";

static const char *__doc_popart_Op_shard_2 = R"doc()doc";

static const char *__doc_popart_Op_shard_3 = R"doc()doc";

static const char *__doc_popart_Op_shard_4 = R"doc()doc";

static const char *__doc_popart_Op_str = R"doc()doc";

static const char *__doc_popart_Op_str_2 = R"doc()doc";

static const char *__doc_popart_Op_subgraphInToOpInIndex = R"doc()doc";

static const char *__doc_popart_Op_subgraphInToOpInIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_subgraphOutToOpOutIndex = R"doc()doc";

static const char *__doc_popart_Op_subgraphOutToOpOutIndex_2 = R"doc()doc";

static const char *__doc_popart_Op_toJSON = R"doc()doc";

static const char *__doc_popart_Op_toJSON_2 = R"doc()doc";

static const char *__doc_popart_Op_unrollShard = R"doc()doc";

static const char *__doc_popart_Op_unrollShard_2 = R"doc()doc";

static const char *__doc_popart_Op_uses = R"doc()doc";

static const char *__doc_popart_Op_uses_2 = R"doc()doc";

static const char *__doc_popart_Optimizer = R"doc(The base Optimizer class)doc";

static const char *__doc_popart_Optimizer_2 =
    R"doc(The base Optimizer class)doc";

static const char *__doc_popart_OptimizerReductionType =
    R"doc(Replicated graph reduction mode (data parallel optimizer). Determines
which replicated collective operations are inserted into the graph.)doc";

static const char *__doc_popart_OptimizerReductionType_2 =
    R"doc(Replicated graph reduction mode (data parallel optimizer). Determines
which replicated collective operations are inserted into the graph.)doc";

static const char *__doc_popart_OptimizerReductionType_AcclReduce =
    R"doc(Momentum reduction (SGD1, every N-th iteration, gradient accumulation))doc";

static const char *__doc_popart_OptimizerReductionType_AcclReduce_2 =
    R"doc(Momentum reduction (SGD1, every N-th iteration, gradient accumulation))doc";

static const char *__doc_popart_OptimizerReductionType_AccumReduce =
    R"doc(Accumulator reduction (Adam, every N-th iteration, gradient
accumulation))doc";

static const char *__doc_popart_OptimizerReductionType_AccumReduce_2 =
    R"doc(Accumulator reduction (Adam, every N-th iteration, gradient
accumulation))doc";

static const char *__doc_popart_OptimizerReductionType_GradReduce =
    R"doc(Gradient reduction (every iteration, after a weight's gradient is
produced))doc";

static const char *__doc_popart_OptimizerReductionType_GradReduce_2 =
    R"doc(Gradient reduction (every iteration, after a weight's gradient is
produced))doc";

static const char *__doc_popart_OptimizerReductionType_None =
    R"doc(No replicated graph reduction)doc";

static const char *__doc_popart_OptimizerReductionType_None_2 =
    R"doc(No replicated graph reduction)doc";

static const char *__doc_popart_OptimizerType = R"doc(Types of optimizers.)doc";

static const char *__doc_popart_OptimizerType_2 =
    R"doc(Types of optimizers.)doc";

static const char *__doc_popart_OptimizerType_Adam = R"doc()doc";

static const char *__doc_popart_OptimizerType_Adam_2 = R"doc()doc";

static const char *__doc_popart_OptimizerType_Adaptive = R"doc()doc";

static const char *__doc_popart_OptimizerType_Adaptive_2 = R"doc()doc";

static const char *__doc_popart_OptimizerType_NTYPES = R"doc()doc";

static const char *__doc_popart_OptimizerType_NTYPES_2 = R"doc()doc";

static const char *__doc_popart_OptimizerType_SGD = R"doc()doc";

static const char *__doc_popart_OptimizerType_SGD_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue =
    R"doc(A class used to represent values of hyper parameters.)doc";

static const char *__doc_popart_OptimizerValue_2 =
    R"doc(A class used to represent values of hyper parameters.)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue =
    R"doc(Equivalent to OptimizerValue(0, false).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_2 =
    R"doc(Equivalent to OptimizerValue(v, true).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_3 =
    R"doc(Constructor.

Parameter ``v``:
    The current value of the hyper parameter.

Parameter ``c``:
    A boolean flag to indicate whether the parameter will remain at
    this value forever (`true`) or may change over time (`false`).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_4 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_5 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_6 =
    R"doc(Equivalent to OptimizerValue(0, false).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_7 =
    R"doc(Equivalent to OptimizerValue(v, true).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_8 =
    R"doc(Constructor.

Parameter ``v``:
    The current value of the hyper parameter.

Parameter ``c``:
    A boolean flag to indicate whether the parameter will remain at
    this value forever (`true`) or may change over time (`false`).)doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_9 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_OptimizerValue_10 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst_3 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_isConst_4 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_operator_assign = R"doc()doc";

static const char *__doc_popart_OptimizerValue_operator_assign_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_operator_eq = R"doc()doc";

static const char *__doc_popart_OptimizerValue_operator_eq_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val_2 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val_3 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_val_4 = R"doc()doc";

static const char *__doc_popart_OptimizerValue_validReplacement = R"doc()doc";

static const char *__doc_popart_OptimizerValue_validReplacement_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer_3 = R"doc()doc";

static const char *__doc_popart_Optimizer_Optimizer_4 = R"doc()doc";

static const char *__doc_popart_Optimizer_accumulationFactor = R"doc()doc";

static const char *__doc_popart_Optimizer_accumulationFactor_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_checkReplacementValue = R"doc()doc";

static const char *__doc_popart_Optimizer_checkReplacementValue_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_clipNormSettings = R"doc()doc";

static const char *__doc_popart_Optimizer_clipNormSettings_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_clone = R"doc()doc";

static const char *__doc_popart_Optimizer_clone_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_createOp = R"doc()doc";

static const char *__doc_popart_Optimizer_createOp_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_enableGradientAccumulation =
    R"doc()doc";

static const char *__doc_popart_Optimizer_enableGradientAccumulation_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_factorsAreSetFromOptions =
    R"doc()doc";

static const char *__doc_popart_Optimizer_factorsAreSetFromOptions_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_getAccumulationFactor = R"doc()doc";

static const char *__doc_popart_Optimizer_getAccumulationFactor_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_getClipNormSettings = R"doc()doc";

static const char *__doc_popart_Optimizer_getClipNormSettings_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_getInputIds = R"doc()doc";

static const char *__doc_popart_Optimizer_getInputIds_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingTensorId = R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingTensorId_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingVal = R"doc()doc";

static const char *__doc_popart_Optimizer_getLossScalingVal_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_getOptimizerInputs = R"doc()doc";

static const char *__doc_popart_Optimizer_getOptimizerInputs_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_getReplicatedGraphCount = R"doc()doc";

static const char *__doc_popart_Optimizer_getReplicatedGraphCount_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_gradientAccumulationEnabled =
    R"doc()doc";

static const char *__doc_popart_Optimizer_gradientAccumulationEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_hash = R"doc()doc";

static const char *__doc_popart_Optimizer_hash_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_lossScaling = R"doc()doc";

static const char *__doc_popart_Optimizer_lossScaling_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_ls = R"doc()doc";

static const char *__doc_popart_Optimizer_ls_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulation =
    R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulation_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulationEnabled =
    R"doc()doc";

static const char *__doc_popart_Optimizer_meanGradientAccumulationEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Optimizer_replicatedGraphCount = R"doc()doc";

static const char *__doc_popart_Optimizer_replicatedGraphCount_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_resetTensorData = R"doc()doc";

static const char *__doc_popart_Optimizer_resetTensorData_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_setFactorsFromOptions = R"doc()doc";

static const char *__doc_popart_Optimizer_setFactorsFromOptions_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_setTensorData = R"doc()doc";

static const char *__doc_popart_Optimizer_setTensorData_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_type = R"doc()doc";

static const char *__doc_popart_Optimizer_type_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_type_s = R"doc()doc";

static const char *__doc_popart_Optimizer_type_s_2 = R"doc()doc";

static const char *__doc_popart_Optimizer_validReplacement = R"doc()doc";

static const char *__doc_popart_Optimizer_validReplacement_2 = R"doc()doc";

static const char *__doc_popart_POpCmp =
    R"doc(To prevent non-determinism, POpCmp is used on any sets and maps that
use pointers to operators as a set/map key.)doc";

static const char *__doc_popart_POpCmp_2 =
    R"doc(To prevent non-determinism, POpCmp is used on any sets and maps that
use pointers to operators as a set/map key.)doc";

static const char *__doc_popart_POpCmp_operator_call = R"doc()doc";

static const char *__doc_popart_POpCmp_operator_call_2 = R"doc()doc";

static const char *__doc_popart_POpIntCmp = R"doc()doc";

static const char *__doc_popart_POpIntCmp_2 = R"doc()doc";

static const char *__doc_popart_POpIntCmp_operator_call = R"doc()doc";

static const char *__doc_popart_POpIntCmp_operator_call_2 = R"doc()doc";

static const char *__doc_popart_PTensorCmp = R"doc()doc";

static const char *__doc_popart_PTensorCmp_operator_call = R"doc()doc";

static const char *__doc_popart_PatternCreator = R"doc()doc";

static const char *__doc_popart_PatternCreator_2 = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator_2 = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator_3 = R"doc()doc";

static const char *__doc_popart_PatternCreator_PatternCreator_4 = R"doc()doc";

static const char *__doc_popart_PatternNames = R"doc()doc";

static const char *__doc_popart_PatternNames_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_addName = R"doc()doc";

static const char *__doc_popart_PatternNames_addName_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_contains = R"doc()doc";

static const char *__doc_popart_PatternNames_contains_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_getInstance = R"doc()doc";

static const char *__doc_popart_PatternNames_getInstance_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_getName = R"doc()doc";

static const char *__doc_popart_PatternNames_getName_2 = R"doc()doc";

static const char *__doc_popart_PatternNames_getName_3 = R"doc()doc";

static const char *__doc_popart_PatternNames_getName_4 = R"doc()doc";

static const char *__doc_popart_PatternNames_names = R"doc()doc";

static const char *__doc_popart_PatternNames_names_2 = R"doc()doc";

static const char *__doc_popart_Patterns = R"doc()doc";

static const char *__doc_popart_Patterns_2 = R"doc()doc";

static const char *__doc_popart_PatternsLevel = R"doc()doc";

static const char *__doc_popart_PatternsLevel_2 = R"doc()doc";

static const char *__doc_popart_PatternsLevel_All = R"doc()doc";

static const char *__doc_popart_PatternsLevel_All_2 = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Default = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Default_2 = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Minimal = R"doc()doc";

static const char *__doc_popart_PatternsLevel_Minimal_2 = R"doc()doc";

static const char *__doc_popart_PatternsLevel_NoPatterns = R"doc()doc";

static const char *__doc_popart_PatternsLevel_NoPatterns_2 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_2 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_3 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_4 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_5 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_6 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_7 = R"doc()doc";

static const char *__doc_popart_Patterns_Patterns_8 = R"doc()doc";

static const char *__doc_popart_Patterns_create = R"doc()doc";

static const char *__doc_popart_Patterns_create_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg0GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableAtan2Arg1GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableCosGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableCosGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableDecomposeBinaryConstScalar =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableDecomposeBinaryConstScalar_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg0GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableDivArg1GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpm1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableExpm1GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableGemmDecomposition = R"doc()doc";

static const char *__doc_popart_Patterns_enableGemmDecomposition_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableInPlace = R"doc()doc";

static const char *__doc_popart_Patterns_enableInPlace_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableInitAccumulate = R"doc()doc";

static const char *__doc_popart_Patterns_enableInitAccumulate_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableLog1pGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableLog1pGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableLogGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableLogGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulLhsGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulLhsGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulRhsGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMatMulRhsGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableMulArgGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableMulArgGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableNegativeOneScale = R"doc()doc";

static const char *__doc_popart_Patterns_enableNegativeOneScale_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableNlllWithSoftMaxGradDirect =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableNlllWithSoftMaxGradDirect_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableOpToIdentity = R"doc()doc";

static const char *__doc_popart_Patterns_enableOpToIdentity_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_3 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_4 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_5 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_6 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_7 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePattern_8 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePostNRepl = R"doc()doc";

static const char *__doc_popart_Patterns_enablePostNRepl_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg0GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg0GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enablePowArg1GradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enablePreUniRepl = R"doc()doc";

static const char *__doc_popart_Patterns_enablePreUniRepl_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomNormalLikeOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomNormalLikeOpPattern_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomUniformLikeOpPattern =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableRandomUniformLikeOpPattern_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableReciprocalGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableReciprocalGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableRuntimeAsserts = R"doc()doc";

static const char *__doc_popart_Patterns_enableRuntimeAsserts_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableSinGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableSinGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableSoftMaxGradDirect = R"doc()doc";

static const char *__doc_popart_Patterns_enableSoftMaxGradDirect_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableSplitGather = R"doc()doc";

static const char *__doc_popart_Patterns_enableSplitGather_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableSqrtGradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableSqrtGradOp_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableSubtractArg1GradOp = R"doc()doc";

static const char *__doc_popart_Patterns_enableSubtractArg1GradOp_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableUpdateInplacePrioritiesForIpu =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableUpdateInplacePrioritiesForIpu_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_enableUpsampleToResize = R"doc()doc";

static const char *__doc_popart_Patterns_enableUpsampleToResize_2 = R"doc()doc";

static const char *__doc_popart_Patterns_enableZerosLikeOpPattern = R"doc()doc";

static const char *__doc_popart_Patterns_enableZerosLikeOpPattern_2 =
    R"doc()doc";

static const char
    *__doc_popart_Patterns_ensureAllMandatoryPreAliasPatternsAreEnabled =
        R"doc()doc";

static const char
    *__doc_popart_Patterns_ensureAllMandatoryPreAliasPatternsAreEnabled_2 =
        R"doc()doc";

static const char *__doc_popart_Patterns_getInplaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_getInplaceEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_getPreAliasList = R"doc()doc";

static const char *__doc_popart_Patterns_getPreAliasList_2 = R"doc()doc";

static const char *__doc_popart_Patterns_getRuntimeAssertsOn = R"doc()doc";

static const char *__doc_popart_Patterns_getRuntimeAssertsOn_2 = R"doc()doc";

static const char *__doc_popart_Patterns_getSettings = R"doc()doc";

static const char *__doc_popart_Patterns_getSettings_2 = R"doc()doc";

static const char *
    __doc_popart_Patterns_getUpdateInplacePrioritiesForIpuEnabled = R"doc()doc";

static const char
    *__doc_popart_Patterns_getUpdateInplacePrioritiesForIpuEnabled_2 =
        R"doc()doc";

static const char *__doc_popart_Patterns_inplaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_inplaceEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg0GradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isAtan2Arg1GradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isCosGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isCosGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isDecomposeBinaryConstScalarEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isDecomposeBinaryConstScalarEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg0GradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isDivArg1GradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isExpGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isExpGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isExpm1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isExpm1GradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isFmodArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isFmodArg0GradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isGemmDecompositionEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isGemmDecompositionEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isInPlaceEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isInPlaceEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isInitAccumulateEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isInitAccumulateEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isLog1pGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isLog1pGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isLogGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isLogGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulLhsGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulLhsGradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulRhsGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMatMulRhsGradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isMulArgGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isMulArgGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isNegativeOneScaleEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isNegativeOneScaleEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isNlllWithSoftMaxGradDirectEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isNlllWithSoftMaxGradDirectEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isOpToIdentityEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isOpToIdentityEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_3 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_4 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_5 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_6 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_7 = R"doc()doc";

static const char *__doc_popart_Patterns_isPatternEnabled_8 = R"doc()doc";

static const char *__doc_popart_Patterns_isPostNReplEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPostNReplEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg0GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg0GradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg1GradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPowArg1GradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isPreUniReplEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isPreUniReplEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isRandomNormalLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isRandomNormalLikeOpPatternEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isRandomUniformLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isRandomUniformLikeOpPatternEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isReciprocalGradOpEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isReciprocalGradOpEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSinGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSinGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isSoftMaxGradDirectEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSoftMaxGradDirectEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSplitGatherEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSplitGatherEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isSqrtGradOpEnabled = R"doc()doc";

static const char *__doc_popart_Patterns_isSqrtGradOpEnabled_2 = R"doc()doc";

static const char *__doc_popart_Patterns_isSubtractArg1GradOpEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isSubtractArg1GradOpEnabled_2 =
    R"doc()doc";

static const char
    *__doc_popart_Patterns_isUpdateInplacePrioritiesForIpuEnabled = R"doc()doc";

static const char
    *__doc_popart_Patterns_isUpdateInplacePrioritiesForIpuEnabled_2 =
        R"doc()doc";

static const char *__doc_popart_Patterns_isUpsampleToResizeEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isUpsampleToResizeEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_isZerosLikeOpPatternEnabled =
    R"doc()doc";

static const char *__doc_popart_Patterns_isZerosLikeOpPatternEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_Patterns_operator_eq = R"doc()doc";

static const char *__doc_popart_Patterns_operator_eq_2 = R"doc()doc";

static const char *__doc_popart_Patterns_runtimeAssertsOn = R"doc()doc";

static const char *__doc_popart_Patterns_runtimeAssertsOn_2 = R"doc()doc";

static const char *__doc_popart_Patterns_settings = R"doc()doc";

static const char *__doc_popart_Patterns_settings_2 = R"doc()doc";

static const char *__doc_popart_Patterns_updateInplacePrioritiesForIpuEnabled =
    R"doc()doc";

static const char
    *__doc_popart_Patterns_updateInplacePrioritiesForIpuEnabled_2 = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_2 = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_2 =
    R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_enabledByDefault =
        R"doc()doc";

static const char *
    __doc_popart_PreAliasPatternManager_PreAliasPatternInfo_enabledByDefault_2 =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_factory =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_factory_2 =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_mandatory =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_mandatory_2 =
        R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_name = R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternInfo_name_2 =
        R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_PreAliasPatternManager =
    R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_PreAliasPatternManager_2 = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_createPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_createPattern_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInfo = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInfo_2 = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInstance =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getInstance_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternInfos =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternInfos_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternName =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getPatternName_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex_3 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_getTypeIndex_4 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_opReplacementPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_opReplacementPattern_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_patternInfos =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_patternInfos_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_patternTypeToTypeIndex =
    R"doc()doc";

static const char
    *__doc_popart_PreAliasPatternManager_patternTypeToTypeIndex_2 = R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern_2 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern_3 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_registerPattern_4 =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_tryGetTypeIndex =
    R"doc()doc";

static const char *__doc_popart_PreAliasPatternManager_tryGetTypeIndex_2 =
    R"doc()doc";

static const char *__doc_popart_RecomputationType =
    R"doc(Enum type to specify which ops to recompute in the backwards pass when
doing auto-recomputation.)doc";

static const char *__doc_popart_RecomputationType_2 =
    R"doc(Enum type to specify which ops to recompute in the backwards pass when
doing auto-recomputation.)doc";

static const char *__doc_popart_RecomputationType_N =
    R"doc(The number of ``RecomputationTypes`` values.)doc";

static const char *__doc_popart_RecomputationType_N_2 =
    R"doc(The number of ``RecomputationTypes`` values.)doc";

static const char *__doc_popart_RecomputationType_None =
    R"doc(No ops are recomputed.)doc";

static const char *__doc_popart_RecomputationType_None_2 =
    R"doc(No ops are recomputed.)doc";

static const char *__doc_popart_RecomputationType_NormOnly =
    R"doc(Only Norm ops (+ non-linearities, if following) are recomputed.)doc";

static const char *__doc_popart_RecomputationType_NormOnly_2 =
    R"doc(Only Norm ops (+ non-linearities, if following) are recomputed.)doc";

static const char *__doc_popart_RecomputationType_Pipeline =
    R"doc(Recompute all forward pipeline stages.)doc";

static const char *__doc_popart_RecomputationType_Pipeline_2 =
    R"doc(Recompute all forward pipeline stages.)doc";

static const char *__doc_popart_RecomputationType_Standard =
    R"doc(Algorithm to pick checkpoints to try and minimise max liveness.)doc";

static const char *__doc_popart_RecomputationType_Standard_2 =
    R"doc(Algorithm to pick checkpoints to try and minimise max liveness.)doc";

static const char *__doc_popart_RecomputeType = R"doc()doc";

static const char *__doc_popart_RecomputeType_2 = R"doc()doc";

static const char *__doc_popart_RecomputeType_3 = R"doc()doc";

static const char *__doc_popart_RecomputeType_4 = R"doc()doc";

static const char *__doc_popart_RecomputeType_Checkpoint = R"doc()doc";

static const char *__doc_popart_RecomputeType_Checkpoint_2 = R"doc()doc";

static const char *__doc_popart_RecomputeType_Recompute = R"doc()doc";

static const char *__doc_popart_RecomputeType_Recompute_2 = R"doc()doc";

static const char *__doc_popart_RecomputeType_Recomputed = R"doc()doc";

static const char *__doc_popart_RecomputeType_Recomputed_2 = R"doc()doc";

static const char *__doc_popart_RecomputeType_Undefined = R"doc()doc";

static const char *__doc_popart_RecomputeType_Undefined_2 = R"doc()doc";

static const char *__doc_popart_ReductionType =
    R"doc(Defines the type of reduction used when weight updates of a batch are
computed in one go and are reduced over the gradients of the whole
minibatch.)doc";

static const char *__doc_popart_ReductionType_2 =
    R"doc(Defines the type of reduction used when weight updates of a batch are
computed in one go and are reduced over the gradients of the whole
minibatch.)doc";

static const char *__doc_popart_ReductionType_Mean =
    R"doc(Take the mean of the loss values and divide the gradient by the number
of samples.)doc";

static const char *__doc_popart_ReductionType_Mean_2 =
    R"doc(Take the mean of the loss values and divide the gradient by the number
of samples.)doc";

static const char *__doc_popart_ReductionType_N =
    R"doc(The number of ReductionType values.)doc";

static const char *__doc_popart_ReductionType_N_2 =
    R"doc(The number of ReductionType values.)doc";

static const char *__doc_popart_ReductionType_NoReduction =
    R"doc(Leave the loss values as they are and do not scale the gradient.)doc";

static const char *__doc_popart_ReductionType_NoReduction_2 =
    R"doc(Leave the loss values as they are and do not scale the gradient.)doc";

static const char *__doc_popart_ReductionType_Sum =
    R"doc(Sum the output of the loss values and do not scale the gradient.)doc";

static const char *__doc_popart_ReductionType_Sum_2 =
    R"doc(Sum the output of the loss values and do not scale the gradient.)doc";

static const char *__doc_popart_RemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_2 = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_RemoteBufferInfo = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_RemoteBufferInfo_2 =
    R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_info = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_info_2 = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_repeats = R"doc()doc";

static const char *__doc_popart_RemoteBufferInfo_repeats_2 = R"doc()doc";

static const char *__doc_popart_ReplicatedTensorSharding =
    R"doc(Enum type to specify whether to shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_2 =
    R"doc(Enum type to specify whether to shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_Off =
    R"doc(Don't shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_Off_2 =
    R"doc(Don't shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_On =
    R"doc(Do shard tensors over replicas.)doc";

static const char *__doc_popart_ReplicatedTensorSharding_On_2 =
    R"doc(Do shard tensors over replicas.)doc";

static const char *__doc_popart_RequireOptimalSchedule = R"doc()doc";

static const char *__doc_popart_RequireOptimalSchedule_2 = R"doc()doc";

static const char *__doc_popart_SGD =
    R"doc(Stochastic Gradient Descent (%SGD) optimizer.

Akin to any optimizer implementation, this class is responsible for
updating each weight tensor ($w$) in the model using the gradient
($g$) of the loss function with respect to the weight as calculated
during the backwards pass.

The %SGD optimizer has the following **state** for each weight:

* *velocity* ($v$)

The %SGD optimizer has the following **hyper parameters**:

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

static const char *__doc_popart_SGD_2 =
    R"doc(Stochastic Gradient Descent (%SGD) optimizer.

Akin to any optimizer implementation, this class is responsible for
updating each weight tensor ($w$) in the model using the gradient
($g$) of the loss function with respect to the weight as calculated
during the backwards pass.

The %SGD optimizer has the following **state** for each weight:

* *velocity* ($v$)

The %SGD optimizer has the following **hyper parameters**:

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
    The learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    The weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultMomentum``:
    The momentum value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultDampening``:
    The dampening value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultVelocityScaling``:
    The velocity scaling value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``lossScaling``:
    The loss scaling value to use.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_SGD_SGD_2 =
    R"doc(Constructor.

Parameter ``params``:
    A parameter map where the keys are one or more of
    `"defaultLearningRate"`, `"defaultWeightDecay"`,
    `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"`. The map's values
    are pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter because default values will be used where parameters are
    missing.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
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

static const char *__doc_popart_SGD_SGD_5 =
    R"doc(Constructor.

Parameter ``defaultLearningRate``:
    The learning rate value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultWeightDecay``:
    The weight decay value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultMomentum``:
    The momentum value to use for weights for which no weight-specific
    hyper parameter have been inserted.

Parameter ``defaultDampening``:
    The dampening value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``defaultVelocityScaling``:
    The velocity scaling value to use for weights for which no weight-
    specific hyper parameter have been inserted.

Parameter ``lossScaling``:
    The loss scaling value to use.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
    values for weights).)doc";

static const char *__doc_popart_SGD_SGD_6 =
    R"doc(Constructor.

Parameter ``params``:
    A parameter map where the keys are one or more of
    `"defaultLearningRate"`, `"defaultWeightDecay"`,
    `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"`. The map's values
    are pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter because default values will be used where parameters are
    missing.

Parameter ``clipNormSettings``:
    A vector of ClipNormSettings (this can be used to set maximum
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

static const char *__doc_popart_SGD_SGD_7 =
    R"doc(Construct an SDG instance with default values.)doc";

static const char *__doc_popart_SGD_SGD_8 = R"doc()doc";

static const char *__doc_popart_SGD_clone = R"doc()doc";

static const char *__doc_popart_SGD_clone_2 = R"doc()doc";

static const char *__doc_popart_SGD_createOp = R"doc()doc";

static const char *__doc_popart_SGD_createOp_2 = R"doc()doc";

static const char *__doc_popart_SGD_dampenings = R"doc()doc";

static const char *__doc_popart_SGD_dampenings_2 = R"doc()doc";

static const char *__doc_popart_SGD_dps = R"doc()doc";

static const char *__doc_popart_SGD_dps_2 = R"doc()doc";

static const char *__doc_popart_SGD_dpsf1helper = R"doc()doc";

static const char *__doc_popart_SGD_dpsf1helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_fromDefaultMap = R"doc()doc";

static const char *__doc_popart_SGD_fromDefaultMap_2 = R"doc()doc";

static const char *__doc_popart_SGD_getComplete = R"doc()doc";

static const char *__doc_popart_SGD_getComplete_2 = R"doc()doc";

static const char *__doc_popart_SGD_getInputIds =
    R"doc(The names of the inputs for the VarUpdateOp for the variable tensor \p
weight. In the returned vector, an empty string ("") is used as a
placeholder for constant inputs.)doc";

static const char *__doc_popart_SGD_getInputIds_2 =
    R"doc(The names of the inputs for the VarUpdateOp for the variable tensor \p
weight. In the returned vector, an empty string ("") is used as a
placeholder for constant inputs.)doc";

static const char *__doc_popart_SGD_getInverseLossScalingTensorId = R"doc()doc";

static const char *__doc_popart_SGD_getInverseLossScalingTensorId_2 =
    R"doc()doc";

static const char *__doc_popart_SGD_getOptimizerInputs =
    R"doc(The names and information for the optimizer tensors.)doc";

static const char *__doc_popart_SGD_getOptimizerInputs_2 =
    R"doc(The names and information for the optimizer tensors.)doc";

static const char *__doc_popart_SGD_getStoredValue =
    R"doc(Tensor "opt" has an id, which it uses to match a compound scalar which
this object can compute from the atomic scalars.)doc";

static const char *__doc_popart_SGD_getStoredValue_2 =
    R"doc(Tensor "opt" has an id, which it uses to match a compound scalar which
this object can compute from the atomic scalars.)doc";

static const char *__doc_popart_SGD_getUnsetDampening =
    R"doc(Default dampening value.)doc";

static const char *__doc_popart_SGD_getUnsetDampening_2 =
    R"doc(Default dampening value.)doc";

static const char *__doc_popart_SGD_getUnsetLearningRate =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_SGD_getUnsetLearningRate_2 =
    R"doc(Default learning rate value.)doc";

static const char *__doc_popart_SGD_getUnsetLossScaling =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetLossScaling_2 =
    R"doc(Default loss scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetMomentum =
    R"doc(Default momentum value.)doc";

static const char *__doc_popart_SGD_getUnsetMomentum_2 =
    R"doc(Default momentum value.)doc";

static const char *__doc_popart_SGD_getUnsetVelocityScaling =
    R"doc(Default velocity scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetVelocityScaling_2 =
    R"doc(Default velocity scaling value.)doc";

static const char *__doc_popart_SGD_getUnsetWeightDecay =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_SGD_getUnsetWeightDecay_2 =
    R"doc(Default weight decay value.)doc";

static const char *__doc_popart_SGD_hasSpecific = R"doc()doc";

static const char *__doc_popart_SGD_hasSpecific_2 = R"doc()doc";

static const char *__doc_popart_SGD_hasSpecific_3 = R"doc()doc";

static const char *__doc_popart_SGD_hasSpecific_4 = R"doc()doc";

static const char *__doc_popart_SGD_hash = R"doc()doc";

static const char *__doc_popart_SGD_hash_2 = R"doc()doc";

static const char *__doc_popart_SGD_insertSpecific =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``learningRate``:
    The learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    The weight decay value to use for this specific weight.

Parameter ``momentum``:
    The momentum value to use for this specific weight.

Parameter ``dampening``:
    The dampening value to use for this specific weight.

Parameter ``velocityScaling``:
    The velocity scaling value to use for this specific weight.)doc";

static const char *__doc_popart_SGD_insertSpecific_2 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"` and the map's values
    pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_SGD_insertSpecific_3 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``learningRate``:
    The learning rate value to use for this specific weight.

Parameter ``weightDecay``:
    The weight decay value to use for this specific weight.

Parameter ``momentum``:
    The momentum value to use for this specific weight.

Parameter ``dampening``:
    The dampening value to use for this specific weight.

Parameter ``velocityScaling``:
    The velocity scaling value to use for this specific weight.)doc";

static const char *__doc_popart_SGD_insertSpecific_4 =
    R"doc(Insert a weight-specific set of hyper parameters.

Parameter ``weight``:
    The TensorId of the weight.

Parameter ``params``:
    A parameter map where keys are one of `"defaultLearningRate"`,
    `"defaultWeightDecay"`, `"defaultMomentum"`, `"defaultDampening"`,
    `"defaultVelocityScaling"` or `"lossScaling"` and the map's values
    pairs of floats and booleans representing OptimizerValue
    constructor arguments. The map does not have to specify each hyper
    parameter as default values will be used where parameters are
    missing.)doc";

static const char *__doc_popart_SGD_learningRates = R"doc()doc";

static const char *__doc_popart_SGD_learningRates_2 = R"doc()doc";

static const char *__doc_popart_SGD_lrs = R"doc()doc";

static const char *__doc_popart_SGD_lrs_2 = R"doc()doc";

static const char *__doc_popart_SGD_mms = R"doc()doc";

static const char *__doc_popart_SGD_mms_2 = R"doc()doc";

static const char *__doc_popart_SGD_momentums = R"doc()doc";

static const char *__doc_popart_SGD_momentums_2 = R"doc()doc";

static const char *__doc_popart_SGD_requiresAccl =
    R"doc(If velocity (accumulation) is required, either because of gradient
accumulation or because of momentum, then return true otherwise return
false.)doc";

static const char *__doc_popart_SGD_requiresAccl_2 =
    R"doc(If velocity (accumulation) is required, either because of gradient
accumulation or because of momentum, then return true otherwise return
false.)doc";

static const char *__doc_popart_SGD_resetTensorData = R"doc()doc";

static const char *__doc_popart_SGD_resetTensorData_2 = R"doc()doc";

static const char *__doc_popart_SGD_runValueChecks = R"doc()doc";

static const char *__doc_popart_SGD_runValueChecks_2 = R"doc()doc";

static const char *__doc_popart_SGD_setTensorData = R"doc()doc";

static const char *__doc_popart_SGD_setTensorData_2 = R"doc()doc";

static const char *__doc_popart_SGD_slr0helper = R"doc()doc";

static const char *__doc_popart_SGD_slr0helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_slr1helper = R"doc()doc";

static const char *__doc_popart_SGD_slr1helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_smm1helper = R"doc()doc";

static const char *__doc_popart_SGD_smm1helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_swd1helper = R"doc()doc";

static const char *__doc_popart_SGD_swd1helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_type = R"doc()doc";

static const char *__doc_popart_SGD_type_2 = R"doc()doc";

static const char *__doc_popart_SGD_type_s = R"doc()doc";

static const char *__doc_popart_SGD_type_s_2 = R"doc()doc";

static const char *__doc_popart_SGD_validReplacement = R"doc()doc";

static const char *__doc_popart_SGD_validReplacement_2 = R"doc()doc";

static const char *__doc_popart_SGD_velocityScalings = R"doc()doc";

static const char *__doc_popart_SGD_velocityScalings_2 = R"doc()doc";

static const char *__doc_popart_SGD_vss = R"doc()doc";

static const char *__doc_popart_SGD_vss_2 = R"doc()doc";

static const char *__doc_popart_SGD_wds = R"doc()doc";

static const char *__doc_popart_SGD_wds_2 = R"doc()doc";

static const char *__doc_popart_SGD_wdsf0helper = R"doc()doc";

static const char *__doc_popart_SGD_wdsf0helper_2 = R"doc()doc";

static const char *__doc_popart_SGD_weightDecays = R"doc()doc";

static const char *__doc_popart_SGD_weightDecays_2 = R"doc()doc";

static const char *__doc_popart_Session =
    R"doc(Session is a runtime instance that provides an interface for executing
ONNX graphs on IPU hardware.)doc";

static const char *__doc_popart_Session_2 =
    R"doc(Session is a runtime instance that provides an interface for executing
ONNX graphs on IPU hardware.)doc";

static const char *__doc_popart_SessionOptions = R"doc()doc";

static const char *__doc_popart_SessionOptions_2 = R"doc()doc";

static const char *__doc_popart_SessionOptions_3 = R"doc()doc";

static const char *__doc_popart_SessionOptions_4 = R"doc()doc";

static const char *__doc_popart_SessionOptions_5 =
    R"doc(A structure containing user configuration options for the Session
class.)doc";

static const char *__doc_popart_SessionOptions_6 =
    R"doc(A structure containing user configuration options for the Session
class.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles =
    R"doc(A wrapper class for the #numIOTiles option that permits any int value
and has an 'unassigned' state.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_2 =
    R"doc(A wrapper class for the #numIOTiles option that permits any int value
and has an 'unassigned' state.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles_2 =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles_3 =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_NumIOTiles_4 =
    R"doc(Constructor.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_assign =
    R"doc(Assign value using int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_assign_2 =
    R"doc(Assign value using int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_eq =
    R"doc(Compare with int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_eq_2 =
    R"doc(Compare with int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_int =
    R"doc(Auto convert to int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_operator_int_2 =
    R"doc(Auto convert to int.)doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_userAssignedValue =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_userAssignedValue_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_value = R"doc()doc";

static const char *__doc_popart_SessionOptions_NumIOTiles_value_2 = R"doc()doc";

static const char *__doc_popart_SessionOptions_accumulateOuterFragmentSettings =
    R"doc(Configuration setting for operations in the accumulate outer fragment.)doc";

static const char *__doc_popart_SessionOptions_accumulateOuterFragmentSettings_2 =
    R"doc(Configuration setting for operations in the accumulate outer fragment.)doc";

static const char
    *__doc_popart_SessionOptions_accumulationAndReplicationReductionType =
        R"doc(Specify how gradients are reduced when using gradient accumulation and
graph replication. This option replaces ``accumulationReductionType``.)doc";

static const char
    *__doc_popart_SessionOptions_accumulationAndReplicationReductionType_2 =
        R"doc(Specify how gradients are reduced when using gradient accumulation and
graph replication. This option replaces ``accumulationReductionType``.)doc";

static const char *__doc_popart_SessionOptions_accumulationFactor =
    R"doc(Specify the number of micro-batches to accumulate before applying the
varUpdate.)doc";

static const char *__doc_popart_SessionOptions_accumulationFactor_2 =
    R"doc(Specify the number of micro-batches to accumulate before applying the
varUpdate.)doc";

static const char *__doc_popart_SessionOptions_accumulationReductionType =
    R"doc(Specify how gradients are reduced when using gradient accumulation.
Note, this option has been deprecated in favour of
``accumulationAndReplicationReductionType``, and will be removed in a
future release.)doc";

static const char *__doc_popart_SessionOptions_accumulationReductionType_2 =
    R"doc(Specify how gradients are reduced when using gradient accumulation.
Note, this option has been deprecated in favour of
``accumulationAndReplicationReductionType``, and will be removed in a
future release.)doc";

static const char
    *__doc_popart_SessionOptions_accumulatorTensorLocationSettings =
        R"doc(Tensor location for gradient accumulator tensors.)doc";

static const char
    *__doc_popart_SessionOptions_accumulatorTensorLocationSettings_2 =
        R"doc(Tensor location for gradient accumulator tensors.)doc";

static const char
    *__doc_popart_SessionOptions_activationTensorLocationSettings =
        R"doc(Tensor location settings for activation/gradient tensors.)doc";

static const char
    *__doc_popart_SessionOptions_activationTensorLocationSettings_2 =
        R"doc(Tensor location settings for activation/gradient tensors.)doc";

static const char *__doc_popart_SessionOptions_aliasZeroCopy =
    R"doc(Enable zero-copy for subgraphs.)doc";

static const char *__doc_popart_SessionOptions_aliasZeroCopy_2 =
    R"doc(Enable zero-copy for subgraphs.)doc";

static const char *__doc_popart_SessionOptions_autoRecomputation =
    R"doc(Enable recomputation of operations in the graph in the backwards pass
to reduce model size at the cost of computation cycles.)doc";

static const char *__doc_popart_SessionOptions_autoRecomputation_2 =
    R"doc(Enable recomputation of operations in the graph in the backwards pass
to reduce model size at the cost of computation cycles.)doc";

static const char *__doc_popart_SessionOptions_autoRecomputationEnabled =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_autoRecomputationEnabled_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_batchSerializationSettings =
    R"doc(Configuration setting for batch serialization.)doc";

static const char *__doc_popart_SessionOptions_batchSerializationSettings_2 =
    R"doc(Configuration setting for batch serialization.)doc";

static const char *__doc_popart_SessionOptions_cachePath =
    R"doc(Folder to save the ``poplar::Executable`` to.)doc";

static const char *__doc_popart_SessionOptions_cachePath_2 =
    R"doc(Folder to save the ``poplar::Executable`` to.)doc";

static const char *__doc_popart_SessionOptions_compilationProgressLogger =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_compilationProgressLogger_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_compileEngine =
    R"doc(If false, the backend will build the Poplar graph but not compile it
into an Engine. In this case, no execution can be performed, and
nothing can be transferred to the device. API calls which retrieve
information from the graph building stage, such as tile mapping
introspection, can still be used.)doc";

static const char *__doc_popart_SessionOptions_compileEngine_2 =
    R"doc(If false, the backend will build the Poplar graph but not compile it
into an Engine. In this case, no execution can be performed, and
nothing can be transferred to the device. API calls which retrieve
information from the graph building stage, such as tile mapping
introspection, can still be used.)doc";

static const char *__doc_popart_SessionOptions_constantWeights =
    R"doc(An optimization for an inference session to have constant weights,
true by default. Set this option to false if you are going to want to
change the weights with a call to Session::resetHostWeights after the
session has been prepared. This option has no effect on a training
session)doc";

static const char *__doc_popart_SessionOptions_constantWeights_2 =
    R"doc(An optimization for an inference session to have constant weights,
true by default. Set this option to false if you are going to want to
change the weights with a call to Session::resetHostWeights after the
session has been prepared. This option has no effect on a training
session)doc";

static const char *__doc_popart_SessionOptions_convolutionOptions =
    R"doc(Poplar convolution options.)doc";

static const char *__doc_popart_SessionOptions_convolutionOptions_2 =
    R"doc(Poplar convolution options.)doc";

static const char *__doc_popart_SessionOptions_customCodeletCompileFlags =
    R"doc(Compile flags for the custom codelets. For example `-g` to generate
debug info.)doc";

static const char *__doc_popart_SessionOptions_customCodeletCompileFlags_2 =
    R"doc(Compile flags for the custom codelets. For example `-g` to generate
debug info.)doc";

static const char *__doc_popart_SessionOptions_customCodelets =
    R"doc(List of codelets (with filetype) to be added to the Poplar graph. See
the Poplar documentation for more information.)doc";

static const char *__doc_popart_SessionOptions_customCodelets_2 =
    R"doc(List of codelets (with filetype) to be added to the Poplar graph. See
the Poplar documentation for more information.)doc";

static const char *__doc_popart_SessionOptions_decomposeGradSum =
    R"doc(Replaces single sums of partial gradients with a tree of additions.
This can reduce max liveness at the cost of extra cycles. A typical
use case for this would be if a large weight tensor is used as an
input to many operations.)doc";

static const char *__doc_popart_SessionOptions_decomposeGradSum_2 =
    R"doc(Replaces single sums of partial gradients with a tree of additions.
This can reduce max liveness at the cost of extra cycles. A typical
use case for this would be if a large weight tensor is used as an
input to many operations.)doc";

static const char *__doc_popart_SessionOptions_defaultPrefetchBufferingDepth =
    R"doc(When #enablePrefetchDatastreams is set this is the default buffering
depth value used for input streams that are not re-arranged on the
host. This value can be overridden via #prefetchBufferingDepthMap.)doc";

static const char *__doc_popart_SessionOptions_defaultPrefetchBufferingDepth_2 =
    R"doc(When #enablePrefetchDatastreams is set this is the default buffering
depth value used for input streams that are not re-arranged on the
host. This value can be overridden via #prefetchBufferingDepthMap.)doc";

static const char *__doc_popart_SessionOptions_delayVarUpdates =
    R"doc(Options to delay variable updates as much as possible.)doc";

static const char *__doc_popart_SessionOptions_delayVarUpdates_2 =
    R"doc(Options to delay variable updates as much as possible.)doc";

static const char
    *__doc_popart_SessionOptions_disableGradAccumulationTensorStreams =
        R"doc(If true, the weight gradient tensors are not saved off the device when
``devicex``.weightsFromHost() is called. Note: this option is
overridden if #syntheticDataMode is not #SyntheticDataMode::Off.)doc";

static const char
    *__doc_popart_SessionOptions_disableGradAccumulationTensorStreams_2 =
        R"doc(If true, the weight gradient tensors are not saved off the device when
``devicex``.weightsFromHost() is called. Note: this option is
overridden if #syntheticDataMode is not #SyntheticDataMode::Off.)doc";

static const char *__doc_popart_SessionOptions_dotChecks =
    R"doc(When to write `.dot` files during Ir construction.)doc";

static const char *__doc_popart_SessionOptions_dotChecks_2 =
    R"doc(When to write `.dot` files during Ir construction.)doc";

static const char *__doc_popart_SessionOptions_dotOpNames =
    R"doc(Include the Op name in the `.dot` file (the Op type is always
exported).)doc";

static const char *__doc_popart_SessionOptions_dotOpNames_2 =
    R"doc(Include the Op name in the `.dot` file (the Op type is always
exported).)doc";

static const char *__doc_popart_SessionOptions_enableAutomaticLossScaling =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_enableAutomaticLossScaling_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_enableExplicitMainLoops =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_enableDistributedReplicatedGraphs =
    R"doc(Enable training with Poplar replicated graphs across multiple PopART
instances.)doc";

static const char
    *__doc_popart_SessionOptions_enableDistributedReplicatedGraphs_2 =
        R"doc(Enable training with Poplar replicated graphs across multiple PopART
instances.)doc";

static const char *__doc_popart_SessionOptions_enableEngineCaching =
    R"doc(Enable Poplar executable caching.)doc";

static const char *__doc_popart_SessionOptions_enableEngineCaching_2 =
    R"doc(Enable Poplar executable caching.)doc";

static const char *__doc_popart_SessionOptions_enableFloatingPointChecks =
    R"doc(Throw an exception when floating point errors occur.)doc";

static const char *__doc_popart_SessionOptions_enableFloatingPointChecks_2 =
    R"doc(Throw an exception when floating point errors occur.)doc";

static const char *__doc_popart_SessionOptions_enableFullyConnectedPass =
    R"doc(Enable the global #fullyConnectedPass option for matmuls.)doc";

static const char *__doc_popart_SessionOptions_enableFullyConnectedPass_2 =
    R"doc(Enable the global #fullyConnectedPass option for matmuls.)doc";

static const char *__doc_popart_SessionOptions_enableGradientAccumulation =
    R"doc(Enable gradient accumulation.)doc";

static const char *__doc_popart_SessionOptions_enableGradientAccumulation_2 =
    R"doc(Enable gradient accumulation.)doc";

static const char *__doc_popart_SessionOptions_enableGroupedMatmuls =
    R"doc(Enable/disable the grouping of matmuls that are the same shape.)doc";

static const char *__doc_popart_SessionOptions_enableGroupedMatmuls_2 =
    R"doc(Enable/disable the grouping of matmuls that are the same shape.)doc";

static const char *__doc_popart_SessionOptions_enableLoadAndOffloadRNGState =
    R"doc(Allows to load/offload device RNG state from host.)doc";

static const char *__doc_popart_SessionOptions_enableLoadAndOffloadRNGState_2 =
    R"doc(Allows to load/offload device RNG state from host.)doc";

static const char *__doc_popart_SessionOptions_enableNonStableSoftmax =
    R"doc(By default, we use the stable softmax Poplar function. The input
tensor to softmax, _x_, is preprocessed by subtracting max(_x_) from
each element before computing the exponentials, ensuring numerical
stability. If you are sure the inputs to your softmax operations are
small enough to not cause overflow when computing the exponential, you
can enable the non-stable version instead, to increase the speed.)doc";

static const char *__doc_popart_SessionOptions_enableNonStableSoftmax_2 =
    R"doc(By default, we use the stable softmax Poplar function. The input
tensor to softmax, _x_, is preprocessed by subtracting max(_x_) from
each element before computing the exponentials, ensuring numerical
stability. If you are sure the inputs to your softmax operations are
small enough to not cause overflow when computing the exponential, you
can enable the non-stable version instead, to increase the speed.)doc";

static const char *__doc_popart_SessionOptions_enableOutlining =
    R"doc(Identify and extract repeated parts of computational graph into
subgraphs.)doc";

static const char *__doc_popart_SessionOptions_enableOutlining_2 =
    R"doc(Identify and extract repeated parts of computational graph into
subgraphs.)doc";

static const char *__doc_popart_SessionOptions_enableOutliningCopyCostPruning =
    R"doc(When `true` the cost of copying of cached sections should be included
in the outlining cost model.)doc";

static const char *__doc_popart_SessionOptions_enableOutliningCopyCostPruning_2 =
    R"doc(When `true` the cost of copying of cached sections should be included
in the outlining cost model.)doc";

static const char *__doc_popart_SessionOptions_enablePipelining =
    R"doc(Enable pipelining of virtual graphs)doc";

static const char *__doc_popart_SessionOptions_enablePipelining_2 =
    R"doc(Enable pipelining of virtual graphs)doc";

static const char *__doc_popart_SessionOptions_enablePrefetchDatastreams =
    R"doc(By default, we will use prefetching for input data streams. Poplar
will speculatively read data for a stream before is is required to
allow the 'preparation' of the data to occur in parallel with compute.)doc";

static const char *__doc_popart_SessionOptions_enablePrefetchDatastreams_2 =
    R"doc(By default, we will use prefetching for input data streams. Poplar
will speculatively read data for a stream before is is required to
allow the 'preparation' of the data to occur in parallel with compute.)doc";

static const char *__doc_popart_SessionOptions_enableReplicatedGraphs =
    R"doc(Enable replication of graphs.)doc";

static const char *__doc_popart_SessionOptions_enableReplicatedGraphs_2 =
    R"doc(Enable replication of graphs.)doc";

static const char *__doc_popart_SessionOptions_enableSerializedMatmuls =
    R"doc(Enable/disable the serializing of matmuls.)doc";

static const char *__doc_popart_SessionOptions_enableSerializedMatmuls_2 =
    R"doc(Enable/disable the serializing of matmuls.)doc";

static const char *__doc_popart_SessionOptions_enableStableNorm =
    R"doc(If true, computes the mean first and subtracts the activations from it
before computing the variance. The implementation with this flag set
to true is slower than when set to false. The stable version requires
the first order moment to be estimated and applied to the sample set
before the second order central moment is calculated.)doc";

static const char *__doc_popart_SessionOptions_enableStableNorm_2 =
    R"doc(If true, computes the mean first and subtracts the activations from it
before computing the variance. The implementation with this flag set
to true is slower than when set to false. The stable version requires
the first order moment to be estimated and applied to the sample set
before the second order central moment is calculated.)doc";

static const char *__doc_popart_SessionOptions_enableStochasticRounding =
    R"doc(Enable stochastic rounding.)doc";

static const char *__doc_popart_SessionOptions_enableStochasticRounding_2 =
    R"doc(Enable stochastic rounding.)doc";

static const char *__doc_popart_SessionOptions_enableSupportedDataTypeCasting =
    R"doc()doc";

static const char
    *__doc_popart_SessionOptions_enableSupportedDataTypeCasting_2 = R"doc()doc";

static const char *__doc_popart_SessionOptions_engineOptions =
    R"doc(Poplar engine options.)doc";

static const char *__doc_popart_SessionOptions_engineOptions_2 =
    R"doc(Poplar engine options.)doc";

static const char *__doc_popart_SessionOptions_executionPhaseSettings =
    R"doc(Configuration settings for execution phases.)doc";

static const char *__doc_popart_SessionOptions_executionPhaseSettings_2 =
    R"doc(Configuration settings for execution phases.)doc";

static const char *__doc_popart_SessionOptions_explicitRecomputation =
    R"doc(Enable explicit recomputation.)doc";

static const char *__doc_popart_SessionOptions_explicitRecomputation_2 =
    R"doc(Enable explicit recomputation.)doc";

static const char *__doc_popart_SessionOptions_exportPoplarComputationGraph =
    R"doc(Export Poplar computation graph.)doc";

static const char *__doc_popart_SessionOptions_exportPoplarComputationGraph_2 =
    R"doc(Export Poplar computation graph.)doc";

static const char *__doc_popart_SessionOptions_exportPoplarVertexGraph =
    R"doc(Export Poplar vertex graph.)doc";

static const char *__doc_popart_SessionOptions_exportPoplarVertexGraph_2 =
    R"doc(Export Poplar vertex graph.)doc";

static const char *__doc_popart_SessionOptions_finalDotOp =
    R"doc(See #firstDotOp.)doc";

static const char *__doc_popart_SessionOptions_finalDotOp_2 =
    R"doc(See #firstDotOp.)doc";

static const char *__doc_popart_SessionOptions_firstDotOp =
    R"doc(The ops to write to the `.dot` file will be a continuous interval of
the schedule, controlled by firstDotOp and finalDotOp. In particular,
it will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp)).)doc";

static const char *__doc_popart_SessionOptions_firstDotOp_2 =
    R"doc(The ops to write to the `.dot` file will be a continuous interval of
the schedule, controlled by firstDotOp and finalDotOp. In particular,
it will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp)).)doc";

static const char *__doc_popart_SessionOptions_gclOptions =
    R"doc(GCL options)doc";

static const char *__doc_popart_SessionOptions_gclOptions_2 =
    R"doc(GCL options)doc";

static const char *__doc_popart_SessionOptions_getAccumulationReductionType =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_getAccumulationReductionType_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_getGlobalReplicationFactor =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_getGlobalReplicationFactor_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_getPrefetchBufferingDepth =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_getPrefetchBufferingDepth_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_globalReplicaOffset =
    R"doc(The first replica index that this PopART instance is running.)doc";

static const char *__doc_popart_SessionOptions_globalReplicaOffset_2 =
    R"doc(The first replica index that this PopART instance is running.)doc";

static const char *__doc_popart_SessionOptions_globalReplicationFactor =
    R"doc(The total number of replicas in a multi instance replicated graph
training session (this should be left as the default value (1) if
distributed replicated graphs are disabled). This value includes local
replication.)doc";

static const char *__doc_popart_SessionOptions_globalReplicationFactor_2 =
    R"doc(The total number of replicas in a multi instance replicated graph
training session (this should be left as the default value (1) if
distributed replicated graphs are disabled). This value includes local
replication.)doc";

static const char *__doc_popart_SessionOptions_groupHostSync =
    R"doc(Allows to group the streams from host at the beginning and the streams
to host at the end, this trades off sum-liveness efficiency for cycle
efficiency.)doc";

static const char *__doc_popart_SessionOptions_groupHostSync_2 =
    R"doc(Allows to group the streams from host at the beginning and the streams
to host at the end, this trades off sum-liveness efficiency for cycle
efficiency.)doc";

static const char *__doc_popart_SessionOptions_hardwareInstrumentations =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_hardwareInstrumentations_2 =
    R"doc()doc";

static const char *__doc_popart_SessionOptions_hostAllReduce =
    R"doc(Perform AllReduce operation on the host. Only useful for training
session.)doc";

static const char *__doc_popart_SessionOptions_hostAllReduce_2 =
    R"doc(Perform AllReduce operation on the host. Only useful for training
session.)doc";

static const char *__doc_popart_SessionOptions_hostAllReduceRemoteBuffer =
    R"doc(Enable the use of ``poplar::RemoteBuffers`` for hostAllReduce
operations.)doc";

static const char *__doc_popart_SessionOptions_hostAllReduceRemoteBuffer_2 =
    R"doc(Enable the use of ``poplar::RemoteBuffers`` for hostAllReduce
operations.)doc";

static const char *__doc_popart_SessionOptions_hostWeightUpdate =
    R"doc(Perform weight update on the host. Only useful for training session.)doc";

static const char *__doc_popart_SessionOptions_hostWeightUpdate_2 =
    R"doc(Perform weight update on the host. Only useful for training session.)doc";

static const char
    *__doc_popart_SessionOptions_instrumentWithHardwareCycleCounter =
        R"doc(Add instrumentation to your program to count the number of device
cycles (of a single tile, on a single IPU) that your main program
takes to execute. Expect this to have a small detrimental impact on
performance.)doc";

static const char
    *__doc_popart_SessionOptions_instrumentWithHardwareCycleCounter_2 =
        R"doc(Add instrumentation to your program to count the number of device
cycles (of a single tile, on a single IPU) that your main program
takes to execute. Expect this to have a small detrimental impact on
performance.)doc";

static const char *__doc_popart_SessionOptions_kahnTieBreaker =
    R"doc(The initial scheduling is done with Kahn's algorithm. When several Ops
are free to be scheduled, this controls which method is used.)doc";

static const char *__doc_popart_SessionOptions_kahnTieBreaker_2 =
    R"doc(The initial scheduling is done with Kahn's algorithm. When several Ops
are free to be scheduled, this controls which method is used.)doc";

static const char *__doc_popart_SessionOptions_logDir =
    R"doc(A directory for log traces to be written into.)doc";

static const char *__doc_popart_SessionOptions_logDir_2 =
    R"doc(A directory for log traces to be written into.)doc";

static const char *__doc_popart_SessionOptions_looseThresholdAtPeak =
    R"doc(The #MergeVarUpdateType::AutoLoose VarUpdateOp merging algorithm has
an absolute threshold defined by:

``min``(#mergeVarUpdateMemThreshold, ``liveAtPeak`` -
``liveCurrently`` + #looseThresholdAtPeak)

where: * ``liveAtPeak`` is an estimate of the maximum live memory of
the computation; and * ``liveCurrently`` is an estimate of the live
memory where the threshold is being used to determine whether to
schedule or postpone a VarUpdateOp.)doc";

static const char *__doc_popart_SessionOptions_looseThresholdAtPeak_2 =
    R"doc(The #MergeVarUpdateType::AutoLoose VarUpdateOp merging algorithm has
an absolute threshold defined by:

``min``(#mergeVarUpdateMemThreshold, ``liveAtPeak`` -
``liveCurrently`` + #looseThresholdAtPeak)

where: * ``liveAtPeak`` is an estimate of the maximum live memory of
the computation; and * ``liveCurrently`` is an estimate of the live
memory where the threshold is being used to determine whether to
schedule or postpone a VarUpdateOp.)doc";

static const char *__doc_popart_SessionOptions_lstmOptions =
    R"doc(Poplar LSTM options.)doc";

static const char *__doc_popart_SessionOptions_lstmOptions_2 =
    R"doc(Poplar LSTM options.)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdate =
    R"doc(Enable merging of VarUpdates into groups of VarUpdates, by flattening
and concatenating variable tensors and updating tensors.)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdate_2 =
    R"doc(Enable merging of VarUpdates into groups of VarUpdates, by flattening
and concatenating variable tensors and updating tensors.)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdateMemThreshold =
    R"doc(The #MergeVarUpdateType::AutoLoose and #MergeVarUpdateType::AutoTight
VarUpdateOp merging algorithms have a threshold on the total memory of
variable tensors to merge for updating. Defined as total memory in
bytes.)doc";

static const char *__doc_popart_SessionOptions_mergeVarUpdateMemThreshold_2 =
    R"doc(The #MergeVarUpdateType::AutoLoose and #MergeVarUpdateType::AutoTight
VarUpdateOp merging algorithms have a threshold on the total memory of
variable tensors to merge for updating. Defined as total memory in
bytes.)doc";

static const char *__doc_popart_SessionOptions_numIOTiles =
    R"doc(Number of IPU tiles dedicated to IO.)doc";

static const char *__doc_popart_SessionOptions_numIOTiles_2 =
    R"doc(Number of IPU tiles dedicated to IO.)doc";

static const char *__doc_popart_SessionOptions_operator_assign = R"doc()doc";

static const char *__doc_popart_SessionOptions_operator_assign_2 = R"doc()doc";

static const char
    *__doc_popart_SessionOptions_optimizerStateTensorLocationSettings =
        R"doc(Tensor location for optimizer state tensors.)doc";

static const char
    *__doc_popart_SessionOptions_optimizerStateTensorLocationSettings_2 =
        R"doc(Tensor location for optimizer state tensors.)doc";

static const char *__doc_popart_SessionOptions_opxAliasChecking =
    R"doc(Run Opx checks to verify IR tensor aliasing information corresponds to
lowered Poplar tensor aliasing.)doc";

static const char *__doc_popart_SessionOptions_opxAliasChecking_2 =
    R"doc(Run Opx checks to verify IR tensor aliasing information corresponds to
lowered Poplar tensor aliasing.)doc";

static const char *__doc_popart_SessionOptions_opxModifyChecking =
    R"doc(Run Opx checks to verify IR tensor modification information
corresponds to lowered Poplar tensor modifications.)doc";

static const char *__doc_popart_SessionOptions_opxModifyChecking_2 =
    R"doc(Run Opx checks to verify IR tensor modification information
corresponds to lowered Poplar tensor modifications.)doc";

static const char *__doc_popart_SessionOptions_outlineSequenceBreakCost =
    R"doc(The penalty applied to outlining potential sub-graphs if the sub-graph
to be created breaks up a sequence of operations that are more
efficient (for example for overlapping compute and exchange) when
outlined together. Default value is set to ~10 *
Op::getHighSubgraphValue().)doc";

static const char *__doc_popart_SessionOptions_outlineSequenceBreakCost_2 =
    R"doc(The penalty applied to outlining potential sub-graphs if the sub-graph
to be created breaks up a sequence of operations that are more
efficient (for example for overlapping compute and exchange) when
outlined together. Default value is set to ~10 *
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

static const char *__doc_popart_SessionOptions_outlineThreshold_2 =
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
individually with Builder.setPartialsType(). Valid values are
`"float"` and `"half"`. By default, this is not set, so no global
partials type is imposed.)doc";

static const char *__doc_popart_SessionOptions_partialsTypeMatMuls_2 =
    R"doc(Set the partials type globally for matmuls. Can be overridden
individually with Builder.setPartialsType(). Valid values are
`"float"` and `"half"`. By default, this is not set, so no global
partials type is imposed.)doc";

static const char *__doc_popart_SessionOptions_prefetchBufferingDepthMap =
    R"doc(When #enablePrefetchDatastreams is set this mapping can be used to set
tensor-specific buffering depths for tensors that are streamed to the
host (typically input tensors). This buffering depth could be
envisaged as being the size of a circular buffer that feeds data to
Poplar. A buffering depth greater than 1 may improve the performance
due to increased parallelisation but comes at the cost of increasing
the memory footprint. Streams for tensors that have no entry in this
map default to a buffering depth of 1.)doc";

static const char *__doc_popart_SessionOptions_prefetchBufferingDepthMap_2 =
    R"doc(When #enablePrefetchDatastreams is set this mapping can be used to set
tensor-specific buffering depths for tensors that are streamed to the
host (typically input tensors). This buffering depth could be
envisaged as being the size of a circular buffer that feeds data to
Poplar. A buffering depth greater than 1 may improve the performance
due to increased parallelisation but comes at the cost of increasing
the memory footprint. Streams for tensors that have no entry in this
map default to a buffering depth of 1.)doc";

static const char *__doc_popart_SessionOptions_rearrangeAnchorsOnHost =
    R"doc(Before anchor tensors are streamed from device to host, they are not
necessarily arranged in memory as required when they are to be copied
from host stream to host. This can be done on the device or on the
host. Done on host by default to save memory, but often at the expense
of cycles, especially for larger anchor tensors.)doc";

static const char *__doc_popart_SessionOptions_rearrangeAnchorsOnHost_2 =
    R"doc(Before anchor tensors are streamed from device to host, they are not
necessarily arranged in memory as required when they are to be copied
from host stream to host. This can be done on the device or on the
host. Done on host by default to save memory, but often at the expense
of cycles, especially for larger anchor tensors.)doc";

static const char *__doc_popart_SessionOptions_replicatedGraphCount =
    R"doc(If enableReplicatedGraphs is true, ``replicatedGraphCount`` will set
the number of model replications. For example, if your model uses 1
IPU, a ``replicatedGraphCount`` of 2 will use 2 IPUs. If your model is
pipelined across 4 IPUs, a ``replicatedGraphCount`` of 4 will use 16
IPUs total. Therefore, the number of IPUs you request must be a
multiple of ``replicatedGraphCount``. If the training is done across
multiple instances then the ``replicatedGraphCount`` is the number of
replicas for this instance.)doc";

static const char *__doc_popart_SessionOptions_replicatedGraphCount_2 =
    R"doc(If enableReplicatedGraphs is true, ``replicatedGraphCount`` will set
the number of model replications. For example, if your model uses 1
IPU, a ``replicatedGraphCount`` of 2 will use 2 IPUs. If your model is
pipelined across 4 IPUs, a ``replicatedGraphCount`` of 4 will use 16
IPUs total. Therefore, the number of IPUs you request must be a
multiple of ``replicatedGraphCount``. If the training is done across
multiple instances then the ``replicatedGraphCount`` is the number of
replicas for this instance.)doc";

static const char *__doc_popart_SessionOptions_reportOptions =
    R"doc(Poplar reporting options.)doc";

static const char *__doc_popart_SessionOptions_reportOptions_2 =
    R"doc(Poplar reporting options.)doc";

static const char *__doc_popart_SessionOptions_separateCallOpPdfs =
    R"doc(When generating PDFs of IR graphs, create separate PDFs for each
subgraph.)doc";

static const char *__doc_popart_SessionOptions_separateCallOpPdfs_2 =
    R"doc(When generating PDFs of IR graphs, create separate PDFs for each
subgraph.)doc";

static const char
    *__doc_popart_SessionOptions_serializedPoprithmsShiftGraphsDir =
        R"doc(PopART uses Poprithms for scheduling PopART graphs. The Poprithms
graphs created for scheduling can be optionally serialised (written to
file). The string below specified the directory to serialize Poprithms
graphs to. If it is empty, then the graphs will not be serialised. The
names of serialization files will be `poprithms_shift_graph_i.json`
for the lowest non-existing values of `i`. The directory must already
exist, PopART will not create it.)doc";

static const char
    *__doc_popart_SessionOptions_serializedPoprithmsShiftGraphsDir_2 =
        R"doc(PopART uses Poprithms for scheduling PopART graphs. The Poprithms
graphs created for scheduling can be optionally serialised (written to
file). The string below specified the directory to serialize Poprithms
graphs to. If it is empty, then the graphs will not be serialised. The
names of serialization files will be `poprithms_shift_graph_i.json`
for the lowest non-existing values of `i`. The directory must already
exist, PopART will not create it.)doc";

static const char *__doc_popart_SessionOptions_strictOpVersions =
    R"doc(Strict op version checks will throw an error if the exact version of
an op required for the models opset is not supported. Turning this
check off will cause PopART to fall back to the latest implementation
of the op that is supported. Warning, turning off these checks may
cause undefined behaviour.)doc";

static const char *__doc_popart_SessionOptions_strictOpVersions_2 =
    R"doc(Strict op version checks will throw an error if the exact version of
an op required for the models opset is not supported. Turning this
check off will cause PopART to fall back to the latest implementation
of the op that is supported. Warning, turning off these checks may
cause undefined behaviour.)doc";

static const char *__doc_popart_SessionOptions_subgraphCopyingStrategy =
    R"doc(This setting determines how copies for inputs and outputs for
subgraphs are lowered. By setting this value to JustInTime you may
save memory at the cost of fragmenting subgraphs into multiple Poplar
functions. This may be particularly useful when a number of weight
updates are outlined in one subgraph, as it may prevent multiple
weight tensors from being live at the same time inside the subgraph.)doc";

static const char *__doc_popart_SessionOptions_subgraphCopyingStrategy_2 =
    R"doc(This setting determines how copies for inputs and outputs for
subgraphs are lowered. By setting this value to JustInTime you may
save memory at the cost of fragmenting subgraphs into multiple Poplar
functions. This may be particularly useful when a number of weight
updates are outlined in one subgraph, as it may prevent multiple
weight tensors from being live at the same time inside the subgraph.)doc";

static const char *__doc_popart_SessionOptions_swapLimitScheduler =
    R"doc(The maximum number of improving steps allowed by the scheduling
algorithm before a solution must be returned.)doc";

static const char *__doc_popart_SessionOptions_swapLimitScheduler_2 =
    R"doc(The maximum number of improving steps allowed by the scheduling
algorithm before a solution must be returned.)doc";

static const char *__doc_popart_SessionOptions_syntheticDataMode =
    R"doc(Use synthetic data: disable data transfer to/from the host. Set to
#SyntheticDataMode::Off to use real data.)doc";

static const char *__doc_popart_SessionOptions_syntheticDataMode_2 =
    R"doc(Use synthetic data: disable data transfer to/from the host. Set to
#SyntheticDataMode::Off to use real data.)doc";

static const char *__doc_popart_SessionOptions_tensorLocationSettingsOverride =
    R"doc(Override tensor location for specific tensors by setting a
TensorLocation for specific TensorId values.)doc";

static const char
    *__doc_popart_SessionOptions_tensorLocationSettingsOverride_2 =
        R"doc(Override tensor location for specific tensors by setting a
TensorLocation for specific TensorId values.)doc";

static const char *__doc_popart_SessionOptions_timeLimitScheduler =
    R"doc(The maximum allowed time that can be spent searching for a good graph
schedule before a solution must be returned.)doc";

static const char *__doc_popart_SessionOptions_timeLimitScheduler_2 =
    R"doc(The maximum allowed time that can be spent searching for a good graph
schedule before a solution must be returned.)doc";

static const char *__doc_popart_SessionOptions_virtualGraphMode =
    R"doc(This option allows you to place ops on virtual graphs to achieve model
parallelism - either manually using model annotations, or
automatically.)doc";

static const char *__doc_popart_SessionOptions_virtualGraphMode_2 =
    R"doc(This option allows you to place ops on virtual graphs to achieve model
parallelism - either manually using model annotations, or
automatically.)doc";

static const char *__doc_popart_SessionOptions_weightTensorLocationSettings =
    R"doc(Tensor location for weight tensors.)doc";

static const char *__doc_popart_SessionOptions_weightTensorLocationSettings_2 =
    R"doc(Tensor location for weight tensors.)doc";

static const char *__doc_popart_Session_Session = R"doc()doc";

static const char *__doc_popart_Session_Session_2 = R"doc()doc";

static const char *__doc_popart_Session_Session_3 = R"doc()doc";

static const char *__doc_popart_Session_Session_4 = R"doc()doc";

static const char *__doc_popart_Session_assertExecutableLoaded =
    R"doc(Throws an error if there is no executable.)doc";

static const char *__doc_popart_Session_assertExecutableLoaded_2 =
    R"doc(Throws an error if there is no executable.)doc";

static const char *__doc_popart_Session_cacheEntries =
    R"doc(Map of hashes / filenames of cached executables.)doc";

static const char *__doc_popart_Session_cacheEntries_2 =
    R"doc(Map of hashes / filenames of cached executables.)doc";

static const char *__doc_popart_Session_compileAndExport =
    R"doc(Compiles the graph and exports it to the specified path.

This will create a ``snap::Graph`` and compile the
``poplar::Executable`` before exporting the executable and metadata.

Parameter ``filename``:
    Name of the file where the compiled executable and associated
    metadata will be saved.)doc";

static const char *__doc_popart_Session_compileAndExport_2 =
    R"doc(Compiles the graph and exports it to the specified stream.

This will create a ``snap::Graph`` and compile the
``poplar::Executable`` before exporting the executable and metadata.

Parameter ``out``:
    Stream where the compiled executable and associated metadata will
    be written to.)doc";

static const char *__doc_popart_Session_compileAndExport_3 =
    R"doc(Compiles the graph and exports it to the specified path.

This will create a ``snap::Graph`` and compile the
``poplar::Executable`` before exporting the executable and metadata.

Parameter ``filename``:
    Name of the file where the compiled executable and associated
    metadata will be saved.)doc";

static const char *__doc_popart_Session_compileAndExport_4 =
    R"doc(Compiles the graph and exports it to the specified stream.

This will create a ``snap::Graph`` and compile the
``poplar::Executable`` before exporting the executable and metadata.

Parameter ``out``:
    Stream where the compiled executable and associated metadata will
    be written to.)doc";

static const char *__doc_popart_Session_configureFromOnnx = R"doc()doc";

static const char *__doc_popart_Session_configureFromOnnx_2 = R"doc()doc";

static const char *__doc_popart_Session_ctorCommonLogic = R"doc()doc";

static const char *__doc_popart_Session_ctorCommonLogic_2 = R"doc()doc";

static const char *__doc_popart_Session_device =
    R"doc(Implementation of the computation. For the IPU back-end this is where
calls to Poplar are made.)doc";

static const char *__doc_popart_Session_device_2 =
    R"doc(Implementation of the computation. For the IPU back-end this is where
calls to Poplar are made.)doc";

static const char *__doc_popart_Session_deviceInfo =
    R"doc(Information about the device which this session uses.)doc";

static const char *__doc_popart_Session_deviceInfo_2 =
    R"doc(Information about the device which this session uses.)doc";

static const char *__doc_popart_Session_executable =
    R"doc(The final executable which contains all the data, metadata and
configuration parameters necessary to start running the program on the
device.)doc";

static const char *__doc_popart_Session_executable_2 =
    R"doc(The final executable which contains all the data, metadata and
configuration parameters necessary to start running the program on the
device.)doc";

static const char *__doc_popart_Session_getCycleCount =
    R"doc(Copy the cycle count tensor to host from the device.)doc";

static const char *__doc_popart_Session_getCycleCount_2 =
    R"doc(Copy the cycle count tensor to host from the device.)doc";

static const char *__doc_popart_Session_getDevice = R"doc()doc";

static const char *__doc_popart_Session_getDevice_2 = R"doc()doc";

static const char *__doc_popart_Session_getDevice_3 = R"doc()doc";

static const char *__doc_popart_Session_getDevice_4 = R"doc()doc";

static const char *__doc_popart_Session_getExecutable = R"doc()doc";

static const char *__doc_popart_Session_getExecutable_2 = R"doc()doc";

static const char *__doc_popart_Session_getExecutionReport =
    R"doc(Retrieve the execution report from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the prepareDevice() call has been made.

Parameter ``useCbor``:
    Produce a CBOR formatted report.

Parameter ``resetProfile``:
    Resets the execution profile.

Returns:
    A string containing the execution report.)doc";

static const char *__doc_popart_Session_getExecutionReport_2 =
    R"doc(Retrieve the execution report from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the prepareDevice() call has been made.

Parameter ``useCbor``:
    Produce a CBOR formatted report.

Parameter ``resetProfile``:
    Resets the execution profile.

Returns:
    A string containing the execution report.)doc";

static const char *__doc_popart_Session_getGraphReport =
    R"doc(Retrieve the graph report from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the prepareDevice() call has been made.

Parameter ``useCbor``:
    Produce a CBOR formatted report.

Returns:
    A string containing the graph (compilation) report.)doc";

static const char *__doc_popart_Session_getGraphReport_2 =
    R"doc(Retrieve the graph report from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report. By default a JSON format report is
produced.

This may only be called after the prepareDevice() call has been made.

Parameter ``useCbor``:
    Produce a CBOR formatted report.

Returns:
    A string containing the graph (compilation) report.)doc";

static const char *__doc_popart_Session_getReport =
    R"doc(Retrieve the graph report from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report.

This may only be called after the prepareDevice() call has been made.

Returns:
    A the libpva report containing the graph report.)doc";

static const char *__doc_popart_Session_getInfo =
    R"doc(Get the TensorInfo on a Tensor.)doc";

static const char *__doc_popart_Session_getInfo_2 =
    R"doc(Get the TensorInfo on a Tensor.)doc";

static const char *__doc_popart_Session_getIr = R"doc()doc";

static const char *__doc_popart_Session_getIr_2 = R"doc()doc";

static const char *__doc_popart_Session_getIrLowering = R"doc()doc";

static const char *__doc_popart_Session_getIrLowering_2 = R"doc()doc";

static const char *__doc_popart_Session_getRNGState = R"doc()doc";

static const char *__doc_popart_Session_getRNGState_2 = R"doc()doc";

static const char *__doc_popart_Session_getSerializedGraph =
    R"doc(Retrieve the serialized graph from the ``poplar::Engine``.

A JSON format report is produced.

This may only be called after the prepareDevice() call has been made.

Returns:
    A string containing the serialized graph.)doc";

static const char *__doc_popart_Session_getSerializedGraph_2 =
    R"doc(Retrieve the serialized graph from the ``poplar::Engine``.

A JSON format report is produced.

This may only be called after the prepareDevice() call has been made.

Returns:
    A string containing the serialized graph.)doc";

static const char *__doc_popart_Session_getSummaryReport =
    R"doc(Retrieve the summary from from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report.

This may only be called after the prepareDevice() call has been made.

Parameter ``resetProfile``:
    Resets the execution profile.

Returns:
    A string containing the report.)doc";

static const char *__doc_popart_Session_getSummaryReport_2 =
    R"doc(Retrieve the summary from from the ``poplar::Engine``.

The options which were given to the constructor will influence the
information in the report.

This may only be called after the prepareDevice() call has been made.

Parameter ``resetProfile``:
    Resets the execution profile.

Returns:
    A string containing the report.)doc";

static const char *__doc_popart_Session_ir =
    R"doc(Abstraction of the computation. The Ir is where all the compute graph
optimisations, backwards pass construction, re-computation growing
etc. happens.)doc";

static const char *__doc_popart_Session_ir_2 =
    R"doc(Abstraction of the computation. The Ir is where all the compute graph
optimisations, backwards pass construction, re-computation growing
etc. happens.)doc";

static const char *__doc_popart_Session_loadEngineAndConnectStreams =
    R"doc(Load the engine on the device and connect the streams

This will set up the ``poplar::Streams``.

Note: This call is optional. The engine will implicitly be loaded on
the device when required.)doc";

static const char *__doc_popart_Session_loadEngineAndConnectStreams_2 =
    R"doc(Load the engine on the device and connect the streams

This will set up the ``poplar::Streams``.

Note: This call is optional. The engine will implicitly be loaded on
the device when required.)doc";

static const char *__doc_popart_Session_loadExecutableFromFile =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
file. The file must have been created with compileAndExport()

Parameter ``filename``:
    Name of the file to load the executable from.)doc";

static const char *__doc_popart_Session_loadExecutableFromFile_2 =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
file. The file must have been created with compileAndExport()

Parameter ``filename``:
    Name of the file to load the executable from.)doc";

static const char *__doc_popart_Session_loadExecutableFromStream =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
stream. The stream must have been created with compileAndExport()

Parameter ``in``:
    Stream to load the executable from.)doc";

static const char *__doc_popart_Session_loadExecutableFromStream_2 =
    R"doc(Load the ``poplar::Executable`` and the PopART metadata from the given
stream. The stream must have been created with compileAndExport()

Parameter ``in``:
    Stream to load the executable from.)doc";

static const char *__doc_popart_Session_lowering =
    R"doc(Implementation of the lowering of the PopART Ir to the Poplar Graph.)doc";

static const char *__doc_popart_Session_lowering_2 =
    R"doc(Implementation of the lowering of the PopART Ir to the Poplar Graph.)doc";

static const char *__doc_popart_Session_modelToHost =
    R"doc(Write current model to ONNX file.

Parameter ``fn``:
    Path to file. Can be absolute or relative. If you plan to run your
    program in multiple processes simultaneously, you should avoid
    possible race conditions by writing to different files, for
    example by using temporary files.)doc";

static const char *__doc_popart_Session_modelToHost_2 =
    R"doc(Write current model to ONNX file.

Parameter ``fn``:
    Path to file. Can be absolute or relative. If you plan to run your
    program in multiple processes simultaneously, you should avoid
    possible race conditions by writing to different files, for
    example by using temporary files.)doc";

static const char *__doc_popart_Session_prepareDevice =
    R"doc(Prepare the network for execution.

This will create the ``snap::Graph`` and ``poplar::Engine``.

Parameter ``loadEngine``:
    Load the engine and connect the streams once the device is ready.)doc";

static const char *__doc_popart_Session_prepareDevice_2 =
    R"doc(Prepare the network for execution.

This will create the ``snap::Graph`` and ``poplar::Engine``.

Parameter ``loadEngine``:
    Load the engine and connect the streams once the device is ready.)doc";

static const char *__doc_popart_Session_readWeights =
    R"doc(Read the weights. Must have called weightsToHost() first.

The weight data is written to the addresses in ``weightsIo``.out.)doc";

static const char *__doc_popart_Session_readWeights_2 =
    R"doc(Read the weights. Must have called weightsToHost() first.

The weight data is written to the addresses in ``weightsIo``.out.)doc";

static const char *__doc_popart_Session_resetHostWeights =
    R"doc(Reset the weights with the weights in an ONNX model that differs from
the current model only in weights. This only updates the weights on
the host; the user still needs to call weightsFromHost() after this to
update the weights on the device.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``ignoreWeightsInModelWithoutCorrespondingHostWeight``:
    If true, do not error if there are initializers in the ONNX model
    with no corresponding initializer tensor in the session's IR.)doc";

static const char *__doc_popart_Session_resetHostWeights_2 =
    R"doc(Reset the weights with the weights in an ONNX model that differs from
the current model only in weights. This only updates the weights on
the host; the user still needs to call weightsFromHost() after this to
update the weights on the device.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``ignoreWeightsInModelWithoutCorrespondingHostWeight``:
    If true, do not error if there are initializers in the ONNX model
    with no corresponding initializer tensor in the session's IR.)doc";

static const char *__doc_popart_Session_run =
    R"doc(Perform one step.

Read input data from address in ``stepIO``.in. Write the output data
to addresses in ``stepIO``.out.

Parameter ``stepIO``:
    Input and output data.

Parameter ``debugName``:
    Debug string to identify this run in logs.)doc";

static const char *__doc_popart_Session_run_2 =
    R"doc(Perform one step.

Read input data from address in ``stepIO``.in. Write the output data
to addresses in ``stepIO``.out.

Parameter ``stepIO``:
    Input and output data.

Parameter ``debugName``:
    Debug string to identify this run in logs.)doc";

static const char *__doc_popart_Session_runCalled =
    R"doc(Flag to indicate if run() has been called.)doc";

static const char *__doc_popart_Session_runCalled_2 =
    R"doc(Flag to indicate if run() has been called.)doc";

static const char *__doc_popart_Session_serializeIr =
    R"doc(Serizalise the IR graph to a string.

Parameter ``format``:
    The format to use for serializing.)doc";

static const char *__doc_popart_Session_serializeIr_2 =
    R"doc(Serizalise the IR graph to a string.

Parameter ``format``:
    The format to use for serializing.)doc";

static const char *__doc_popart_Session_setDevice =
    R"doc(Select a device type.

Parameter ``deviceInfo``:
    Defines the type of device to work on.)doc";

static const char *__doc_popart_Session_setDevice_2 =
    R"doc(Select a device type.

Parameter ``deviceInfo``:
    Defines the type of device to work on.)doc";

static const char *__doc_popart_Session_setRNGState = R"doc()doc";

static const char *__doc_popart_Session_setRNGState_2 = R"doc()doc";

static const char *__doc_popart_Session_setRandomSeed =
    R"doc(Sets the random number generator seed on all tiles of the device. This
ensures deterministic behaviour of random operations in the graph.

Parameter ``The``:
    seed value.)doc";

static const char *__doc_popart_Session_setRandomSeed_2 =
    R"doc(Sets the random number generator seed on all tiles of the device. This
ensures deterministic behaviour of random operations in the graph.

Parameter ``The``:
    seed value.)doc";

static const char *__doc_popart_Session_tryLoadExecutable =
    R"doc(Attempts to load a serialized executable. If successful then IR
preparation and ``snap::Graph`` compilation are skipped.)doc";

static const char *__doc_popart_Session_tryLoadExecutable_2 =
    R"doc(Attempts to load a serialized executable. If successful then IR
preparation and ``snap::Graph`` compilation are skipped.)doc";

static const char *__doc_popart_Session_updateExternallySavedTensorLocations =
    R"doc(Update the tensor locations of the tensors in the Session's ONNX
model. The new file will be created at this point, and written to when
the ONNX model is saved with a subsequent call to modelToHost.

Parameter ``fromLocation``:
    All externally saved tensors with location fromLocation will have
    their location updated to toLocation.

Parameter ``toLocation``:
    The updated location. Must not already exist.)doc";

static const char *__doc_popart_Session_updateExternallySavedTensorLocations_2 =
    R"doc(Update the tensor locations of the tensors in the Session's ONNX
model. The new file will be created at this point, and written to when
the ONNX model is saved with a subsequent call to modelToHost.

Parameter ``fromLocation``:
    All externally saved tensors with location fromLocation will have
    their location updated to toLocation.

Parameter ``toLocation``:
    The updated location. Must not already exist.)doc";

static const char *__doc_popart_Session_weightsFromHost =
    R"doc(Write weights from host to the device.)doc";

static const char *__doc_popart_Session_weightsFromHost_2 =
    R"doc(Write weights from host to the device.)doc";

static const char *__doc_popart_Session_weightsFromHostCalled =
    R"doc(Flag to indicate if weightsFromHost() has been called)doc";

static const char *__doc_popart_Session_weightsFromHostCalled_2 =
    R"doc(Flag to indicate if weightsFromHost() has been called)doc";

static const char *__doc_popart_Session_weightsToHost =
    R"doc(Copy the weights to host from the device.)doc";

static const char *__doc_popart_Session_weightsToHost_2 =
    R"doc(Copy the weights to host from the device.)doc";

static const char *__doc_popart_Session_writeWeights =
    R"doc(Write the weights. Must call weightsFromHost() after this.

The weight data is written to the addresses in ``weightsIo``.out.)doc";

static const char *__doc_popart_Session_writeWeights_2 =
    R"doc(Write the weights. Must call weightsFromHost() after this.

The weight data is written to the addresses in ``weightsIo``.out.)doc";

static const char *__doc_popart_ShardingPlan = R"doc()doc";

static const char *__doc_popart_ShardingPlan_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_array = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_array_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_offset = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_ArrayInfo_offset_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_StepIOGeneric = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_StepIOGeneric_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_advance = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_advance_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_assertNumElements = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_assertNumElements_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_get = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_get_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_getTensorInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_getTensorInfo_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_in = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_in_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inComplete = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inComplete_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inputsInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_inputsInfo_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_out = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_out_2 = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_outputsInfo = R"doc()doc";

static const char *__doc_popart_StepIOGeneric_outputsInfo_2 = R"doc()doc";

static const char *__doc_popart_StepIOSplitter = R"doc()doc";

static const char *__doc_popart_StepIOSplitter_2 = R"doc()doc";

static const char *__doc_popart_SubgraphCopyingStrategy =
    R"doc(Enum type that describes how copies for inputs and outputs for
subgraphs are lowered. Currently this only affects subgraphs
associated with CallOps.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_2 =
    R"doc(Enum type that describes how copies for inputs and outputs for
subgraphs are lowered. Currently this only affects subgraphs
associated with CallOps.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_JustInTime =
    R"doc(Copy inputs just before they are consumed and copy outputs as soon as
they are produced. With this strategy subgraphs may be lowered into
multiple Poplar functions.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_JustInTime_2 =
    R"doc(Copy inputs just before they are consumed and copy outputs as soon as
they are produced. With this strategy subgraphs may be lowered into
multiple Poplar functions.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_N =
    R"doc(The number of SubgraphCopyingStrategy values.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_N_2 =
    R"doc(The number of SubgraphCopyingStrategy values.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_OnEnterAndExit =
    R"doc(Copy all inputs before the start of the subgraph, copy all outputs
after all ops in the subgraph. With this strategy subgraphs will
always map to a single Poplar function.)doc";

static const char *__doc_popart_SubgraphCopyingStrategy_OnEnterAndExit_2 =
    R"doc(Copy all inputs before the start of the subgraph, copy all outputs
after all ops in the subgraph. With this strategy subgraphs will
always map to a single Poplar function.)doc";

static const char *__doc_popart_SyncPattern = R"doc()doc";

static const char *__doc_popart_SyncPattern_2 = R"doc()doc";

static const char *__doc_popart_SyncPattern_Full = R"doc()doc";

static const char *__doc_popart_SyncPattern_Full_2 = R"doc()doc";

static const char *__doc_popart_SyncPattern_ReplicaAndLadder = R"doc()doc";

static const char *__doc_popart_SyncPattern_ReplicaAndLadder_2 = R"doc()doc";

static const char *__doc_popart_SyncPattern_SinglePipeline = R"doc()doc";

static const char *__doc_popart_SyncPattern_SinglePipeline_2 = R"doc()doc";

static const char *__doc_popart_SyntheticDataMode =
    R"doc(Enum type used to specify the data source for input tensors.)doc";

static const char *__doc_popart_SyntheticDataMode_2 =
    R"doc(Enum type used to specify the data source for input tensors.)doc";

static const char *__doc_popart_SyntheticDataMode_N =
    R"doc(The number of ``SyntheticDataMode`` values.)doc";

static const char *__doc_popart_SyntheticDataMode_N_2 =
    R"doc(The number of ``SyntheticDataMode`` values.)doc";

static const char *__doc_popart_SyntheticDataMode_Off =
    R"doc(Use real data.)doc";

static const char *__doc_popart_SyntheticDataMode_Off_2 =
    R"doc(Use real data.)doc";

static const char *__doc_popart_SyntheticDataMode_RandomNormal =
    R"doc(Input tensors are initialised with distribution ~N(0,1).)doc";

static const char *__doc_popart_SyntheticDataMode_RandomNormal_2 =
    R"doc(Input tensors are initialised with distribution ~N(0,1).)doc";

static const char *__doc_popart_SyntheticDataMode_Zeros =
    R"doc(Input tensors are initialised to all zeros.)doc";

static const char *__doc_popart_SyntheticDataMode_Zeros_2 =
    R"doc(Input tensors are initialised to all zeros.)doc";

static const char *__doc_popart_Tensor = R"doc()doc";

static const char *__doc_popart_TensorData = R"doc()doc";

static const char *__doc_popart_TensorData_2 = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData_2 = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData_3 = R"doc()doc";

static const char *__doc_popart_TensorData_TensorData_4 = R"doc()doc";

static const char *__doc_popart_TensorData_copyDataAs = R"doc()doc";

static const char *__doc_popart_TensorData_copyDataAs_2 = R"doc()doc";

static const char *__doc_popart_TensorData_data = R"doc()doc";

static const char *__doc_popart_TensorData_data_2 = R"doc()doc";

static const char *__doc_popart_TensorData_data_3 = R"doc()doc";

static const char *__doc_popart_TensorData_data_4 = R"doc()doc";

static const char *__doc_popart_TensorData_data_5 = R"doc()doc";

static const char *__doc_popart_TensorData_data_6 = R"doc()doc";

static const char *__doc_popart_TensorData_resetData = R"doc()doc";

static const char *__doc_popart_TensorData_resetData_2 = R"doc()doc";

static const char *__doc_popart_TensorData_resetData_3 = R"doc()doc";

static const char *__doc_popart_TensorData_resetData_4 = R"doc()doc";

static const char *__doc_popart_TensorLocation =
    R"doc(Class that describes the memory characteristics of one or multiple
tensors.

See also: SessionOptions.)doc";

static const char *__doc_popart_TensorLocation_2 =
    R"doc(Class that describes the memory characteristics of one or multiple
tensors.

See also: SessionOptions.)doc";

static const char *__doc_popart_TensorLocationInfo = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_getRemoteBufferInfo =
    R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_isRemote = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_isSharded = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_operator_eq = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_remote = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_remoteBufferInfo =
    R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_setRemote = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_setRemoteBufferInfo =
    R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_setSharded = R"doc()doc";

static const char *__doc_popart_TensorLocationInfo_sharded = R"doc()doc";

static const char *__doc_popart_TensorLocationSettings =
    R"doc(A structure containing user configuration for cache/offloading
settings.)doc";

static const char *__doc_popart_TensorLocationSettings_2 =
    R"doc(A structure containing user configuration for cache/offloading
settings.)doc";

static const char *__doc_popart_TensorLocationSettings_TensorLocationSettings =
    R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_2 = R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_3 = R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_4 = R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_5 = R"doc()doc";

static const char
    *__doc_popart_TensorLocationSettings_TensorLocationSettings_6 = R"doc()doc";

static const char *__doc_popart_TensorLocationSettings_location =
    R"doc(The default tensor location for this tensor type.)doc";

static const char *__doc_popart_TensorLocationSettings_location_2 =
    R"doc(The default tensor location for this tensor type.)doc";

static const char *__doc_popart_TensorLocationSettings_minElementsForOffChip =
    R"doc(A minimum number of elements below which offloading won't be
considered.)doc";

static const char *__doc_popart_TensorLocationSettings_minElementsForOffChip_2 =
    R"doc(A minimum number of elements below which offloading won't be
considered.)doc";

static const char *
    __doc_popart_TensorLocationSettings_minElementsForReplicatedTensorSharding =
        R"doc(A minimum number of elements below which replicated tensor sharding
(RTS) won't be considered.)doc";

static const char *
    __doc_popart_TensorLocationSettings_minElementsForReplicatedTensorSharding_2 =
        R"doc(A minimum number of elements below which replicated tensor sharding
(RTS) won't be considered.)doc";

static const char *__doc_popart_TensorLocationSettings_operator_assign =
    R"doc()doc";

static const char *__doc_popart_TensorLocationSettings_operator_assign_2 =
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
    The memory location of the tensor(s).

Parameter ``loadTileSet``:
    The tiles through which the tensor(s) are loaded onto the chip.

Parameter ``storageTileSet``:
    The tiles on which the tensor(s) are stored.

Parameter ``replicatedTensorSharding``:
    Whether to apply replicated tensor. sharding.)doc";

static const char *__doc_popart_TensorLocation_TensorLocation_5 = R"doc()doc";

static const char *__doc_popart_TensorLocation_TensorLocation_6 =
    R"doc(Equivalent to calling TensorLocation(TensorStorage::Undefined,
TileSet::Compute, TileSet::Compute, ReplicatedTensorSharding::Off))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_7 =
    R"doc(Equivalent to calling TensorLocation(storage, TileSet::Compute,
TileSet::Compute, ReplicatedTensorSharding::Off))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_8 =
    R"doc(Equivalent to calling TensorLocation(storage, TileSet::Compute,
TileSet::Compute, replicatedTensorSharding))doc";

static const char *__doc_popart_TensorLocation_TensorLocation_9 =
    R"doc(Construct a TensorLocation from parameters.

Parameter ``storage``:
    The memory location of the tensor(s).

Parameter ``loadTileSet``:
    The tiles through which the tensor(s) are loaded onto the chip.

Parameter ``storageTileSet``:
    The tiles on which the tensor(s) are stored.

Parameter ``replicatedTensorSharding``:
    Whether to apply replicated tensor. sharding.)doc";

static const char *__doc_popart_TensorLocation_TensorLocation_10 = R"doc()doc";

static const char *__doc_popart_TensorLocation_isRemote = R"doc()doc";

static const char *__doc_popart_TensorLocation_isRemote_2 = R"doc()doc";

static const char *__doc_popart_TensorLocation_loadTileSet =
    R"doc(The tiles through which the tensor(s) are loaded onto the chip.)doc";

static const char *__doc_popart_TensorLocation_loadTileSet_2 =
    R"doc(The tiles through which the tensor(s) are loaded onto the chip.)doc";

static const char *__doc_popart_TensorLocation_operator_assign = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_assign_2 = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_eq = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_eq_2 = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_ne = R"doc()doc";

static const char *__doc_popart_TensorLocation_operator_ne_2 = R"doc()doc";

static const char *__doc_popart_TensorLocation_replicatedTensorSharding =
    R"doc(Whether to apply replicated tensor sharding (RTS) or not.)doc";

static const char *__doc_popart_TensorLocation_replicatedTensorSharding_2 =
    R"doc(Whether to apply replicated tensor sharding (RTS) or not.)doc";

static const char *__doc_popart_TensorLocation_serialize = R"doc()doc";

static const char *__doc_popart_TensorLocation_serialize_2 = R"doc()doc";

static const char *__doc_popart_TensorLocation_storage =
    R"doc(The memory location of the tensor(s).)doc";

static const char *__doc_popart_TensorLocation_storage_2 =
    R"doc(The memory location of the tensor(s).)doc";

static const char *__doc_popart_TensorLocation_storageTileSet =
    R"doc(The tiles on which the tensor(s) are stored.)doc";

static const char *__doc_popart_TensorLocation_storageTileSet_2 =
    R"doc(The tiles on which the tensor(s) are stored.)doc";

static const char *__doc_popart_TensorStorage =
    R"doc(Enum type that determines where a tensor is stored.)doc";

static const char *__doc_popart_TensorStorage_2 =
    R"doc(Enum type that determines where a tensor is stored.)doc";

static const char *__doc_popart_TensorStorage_OffChip =
    R"doc(Store the tensor in streaming memory.)doc";

static const char *__doc_popart_TensorStorage_OffChip_2 =
    R"doc(Store the tensor in streaming memory.)doc";

static const char *__doc_popart_TensorStorage_OnChip =
    R"doc(Store the tensor in on-chip memory.)doc";

static const char *__doc_popart_TensorStorage_OnChip_2 =
    R"doc(Store the tensor in on-chip memory.)doc";

static const char *__doc_popart_TensorStorage_Undefined =
    R"doc(Location unspecified.)doc";

static const char *__doc_popart_TensorStorage_Undefined_2 =
    R"doc(Location unspecified.)doc";

static const char *__doc_popart_TensorType = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo_TensorTypeInfo = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo_tensorType = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo_tensor_type = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo_type = R"doc()doc";

static const char *__doc_popart_TensorTypeInfo_type_s = R"doc()doc";

static const char *__doc_popart_TensorType_ActGrad = R"doc()doc";

static const char *__doc_popart_TensorType_Const = R"doc()doc";

static const char *__doc_popart_TensorType_N = R"doc()doc";

static const char *__doc_popart_TensorType_Stream = R"doc()doc";

static const char *__doc_popart_TensorType_Unknown = R"doc()doc";

static const char *__doc_popart_TensorType_Variable = R"doc()doc";

static const char *__doc_popart_Tensor_ReplicatedStreamMode = R"doc()doc";

static const char *__doc_popart_Tensor_ReplicatedStreamMode_Broadcast =
    R"doc()doc";

static const char *__doc_popart_Tensor_ReplicatedStreamMode_Replicate =
    R"doc()doc";

static const char *__doc_popart_Tensor_Tensor = R"doc()doc";

static const char *__doc_popart_Tensor_anyAlias = R"doc()doc";

static const char *__doc_popart_Tensor_associatedOps = R"doc()doc";

static const char *__doc_popart_Tensor_clone = R"doc()doc";

static const char *__doc_popart_Tensor_consumers = R"doc()doc";

static const char *__doc_popart_Tensor_consumersAllPreLoss = R"doc()doc";

static const char *__doc_popart_Tensor_data = R"doc()doc";

static const char *__doc_popart_Tensor_di = R"doc()doc";

static const char *__doc_popart_Tensor_getBatchAxis = R"doc()doc";

static const char *__doc_popart_Tensor_getBatchAxisFromOp = R"doc()doc";

static const char *__doc_popart_Tensor_getDataViaGraphTraversal = R"doc()doc";

static const char *__doc_popart_Tensor_getDebugInfo = R"doc()doc";

static const char *__doc_popart_Tensor_getGraph = R"doc()doc";

static const char *__doc_popart_Tensor_getGraph_2 = R"doc()doc";

static const char *__doc_popart_Tensor_getGraphInputIndex = R"doc()doc";

static const char *__doc_popart_Tensor_getGraphOutputIndex = R"doc()doc";

static const char *__doc_popart_Tensor_getIr = R"doc()doc";

static const char *__doc_popart_Tensor_getIr_2 = R"doc()doc";

static const char *__doc_popart_Tensor_getPipelineStages = R"doc()doc";

static const char *__doc_popart_Tensor_getProducer = R"doc()doc";

static const char *__doc_popart_Tensor_getProducerUnsafe = R"doc()doc";

static const char *__doc_popart_Tensor_getReplicatedStreamMode = R"doc()doc";

static const char *__doc_popart_Tensor_getTensorTypeInfo = R"doc()doc";

static const char *__doc_popart_Tensor_getVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Tensor_getVirtualGraphIdAndTileSet =
    R"doc()doc";

static const char *__doc_popart_Tensor_getVirtualGraphIdAndTileSetUnsafe =
    R"doc()doc";

static const char *__doc_popart_Tensor_getVirtualGraphIdUnsafe = R"doc()doc";

static const char *__doc_popart_Tensor_graph = R"doc()doc";

static const char *__doc_popart_Tensor_hasProducer = R"doc()doc";

static const char *__doc_popart_Tensor_hasTensorData = R"doc()doc";

static const char *__doc_popart_Tensor_hasVirtualGraphId = R"doc()doc";

static const char *__doc_popart_Tensor_id = R"doc()doc";

static const char *__doc_popart_Tensor_info = R"doc()doc";

static const char *__doc_popart_Tensor_isAccumulatorTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isAliased = R"doc()doc";

static const char *__doc_popart_Tensor_isAnchored = R"doc()doc";

static const char *__doc_popart_Tensor_isCheckpointTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isExplicitLoopInput = R"doc()doc";

static const char *__doc_popart_Tensor_isGraphInput = R"doc()doc";

static const char *__doc_popart_Tensor_isGraphOutput = R"doc()doc";

static const char *__doc_popart_Tensor_isImplicitLoopInput = R"doc()doc";

static const char *__doc_popart_Tensor_isImplicitRecomputeTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isLoopInput = R"doc()doc";

static const char *__doc_popart_Tensor_isModified = R"doc()doc";

static const char *__doc_popart_Tensor_isOptimizerStateTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isOptimizerTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isRandomSeedTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isRemoteArgTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isRestoreInplaceTensor = R"doc()doc";

static const char *__doc_popart_Tensor_isUnmodifiable = R"doc()doc";

static const char *__doc_popart_Tensor_isWeightTensor = R"doc()doc";

static const char *__doc_popart_Tensor_producer = R"doc()doc";

static const char *__doc_popart_Tensor_replicatedStreamMode = R"doc()doc";

static const char *__doc_popart_Tensor_resetProducer = R"doc()doc";

static const char *__doc_popart_Tensor_setProducer = R"doc()doc";

static const char *__doc_popart_Tensor_setReplicatedStreamMode = R"doc()doc";

static const char *__doc_popart_Tensor_setTensorData = R"doc()doc";

static const char *__doc_popart_Tensor_setTensorType = R"doc()doc";

static const char *__doc_popart_Tensor_str = R"doc()doc";

static const char *__doc_popart_Tensor_tensorData = R"doc()doc";

static const char *__doc_popart_Tensor_tensorData_2 = R"doc()doc";

static const char *__doc_popart_Tensor_tensorLocationInfo = R"doc()doc";

static const char *__doc_popart_Tensor_tensorType = R"doc()doc";

static const char *__doc_popart_Tensor_tensorTypeInfo = R"doc()doc";

static const char *__doc_popart_Tensor_tensor_type = R"doc()doc";

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

static const char *__doc_popart_Tensors_n = R"doc()doc";

static const char *__doc_popart_Tensors_remove = R"doc()doc";

static const char *__doc_popart_Tensors_removeIsolated = R"doc()doc";

static const char *__doc_popart_Tensors_updateAliases = R"doc()doc";

static const char *__doc_popart_TileSet =
    R"doc(Enum type to specify a set of tiles.)doc";

static const char *__doc_popart_TileSet_2 =
    R"doc(Enum type to specify a set of tiles.)doc";

static const char *__doc_popart_TileSet_Compute =
    R"doc(The set of tiles designated for compute operations.)doc";

static const char *__doc_popart_TileSet_Compute_2 =
    R"doc(The set of tiles designated for compute operations.)doc";

static const char *__doc_popart_TileSet_IO =
    R"doc(The set of tiles designated for IO operations.)doc";

static const char *__doc_popart_TileSet_IO_2 =
    R"doc(The set of tiles designated for IO operations.)doc";

static const char *__doc_popart_TrainingSession = R"doc()doc";

static const char *__doc_popart_TrainingSession_2 = R"doc()doc";

static const char *__doc_popart_TrainingSession_connectStreamToCallback =
    R"doc(Connect Poplar stream callbacks. In conjunction with
`getGradAndVarStreamIds` the streams can be used to copy gradients to
the host to perform collective operations after which the variables
can be streamed back after they have been updated to the device. \p
index referes to the replica index when using replicated graphs.)doc";

static const char *__doc_popart_TrainingSession_connectStreamToCallback_2 =
    R"doc(Connect Poplar stream callbacks. In conjunction with
`getGradAndVarStreamIds` the streams can be used to copy gradients to
the host to perform collective operations after which the variables
can be streamed back after they have been updated to the device. \p
index referes to the replica index when using replicated graphs.)doc";

static const char *__doc_popart_TrainingSession_copyFromRemoteBuffer =
    R"doc(Read from a RemoteBuffer object into a user space pointer \p w. This
can be useful when we run larger models with host side reductions
since HEXOPT is currently limited to 128 MB.)doc";

static const char *__doc_popart_TrainingSession_copyFromRemoteBuffer_2 =
    R"doc(Read from a RemoteBuffer object into a user space pointer \p w. This
can be useful when we run larger models with host side reductions
since HEXOPT is currently limited to 128 MB.)doc";

static const char *__doc_popart_TrainingSession_copyToRemoteBuffer =
    R"doc(Write to a RemoteBuffer object from a user space pointer \p w. This
can be useful when we run larger models with host side reductions
since HEXOPT is currently limited to 128 MB.)doc";

static const char *__doc_popart_TrainingSession_copyToRemoteBuffer_2 =
    R"doc(Write to a RemoteBuffer object from a user space pointer \p w. This
can be useful when we run larger models with host side reductions
since HEXOPT is currently limited to 128 MB.)doc";

static const char *__doc_popart_TrainingSession_createFromIr = R"doc()doc";

static const char *__doc_popart_TrainingSession_createFromIr_2 = R"doc()doc";

static const char *__doc_popart_TrainingSession_createFromOnnxModel =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for training.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors.

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches.

Parameter ``loss``:
    The TensorId of the final scalar loss tensor for training.

Parameter ``optimizer``:
    The name of an optimizer to use when training.

Parameter ``userOptions``:
    String to configure session options.

Parameter ``patterns``:
    Optimization patterns to apply.)doc";

static const char *__doc_popart_TrainingSession_createFromOnnxModel_2 =
    R"doc(Create a runtime class for executing an ONNX graph on a set of IPU
hardware for training.

Parameter ``model``:
    Either an ONNX model protobuf, or the name of a file containing an
    ONNX model protobuf.

Parameter ``inputShapeInfo``:
    Information about the shapes of input and output tensors.

Parameter ``dataFlow``:
    Configuration for the data feeds and fetches.

Parameter ``loss``:
    The TensorId of the final scalar loss tensor for training.

Parameter ``optimizer``:
    The name of an optimizer to use when training.

Parameter ``userOptions``:
    String to configure session options.

Parameter ``patterns``:
    Optimization patterns to apply.)doc";

static const char *__doc_popart_TrainingSession_getHostReduceRemoteBuffers =
    R"doc(Access the remote buffers associated with gradient and weight streams
that are used in host side all reduce operations. Only populated if
``hostAllReduce`` and ``hostAllReduceRemoteBuffer`` are enabled.)doc";

static const char *__doc_popart_TrainingSession_getHostReduceRemoteBuffers_2 =
    R"doc(Access the remote buffers associated with gradient and weight streams
that are used in host side all reduce operations. Only populated if
``hostAllReduce`` and ``hostAllReduceRemoteBuffer`` are enabled.)doc";

static const char *__doc_popart_TrainingSession_getHostReduceStreamIds =
    R"doc(Access the stream IDs for variables that are involved in host side
reductions on the host. Only populated if ``hostAllReduce`` is enabled
in the SessionOptions)doc";

static const char *__doc_popart_TrainingSession_getHostReduceStreamIds_2 =
    R"doc(Access the stream IDs for variables that are involved in host side
reductions on the host. Only populated if ``hostAllReduce`` is enabled
in the SessionOptions)doc";

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

static const char *__doc_popart_TrainingSession_updateOptimizerFromHost_2 =
    R"doc(Update the optimizer and the associated hyperparameters but not the
optimizer state tensors.

**NOTE**: The optimizer parameter has to be compatible with the
optimizer passed to the constructor. For example, you cannot call this
function with an SDG1 optimizer if you created the session with an
SDG0 optimizer. The reason for this is that it is not possible to
change the IR after it has been constructed.

Parameter ``optimizer``:
    A pointer to a popart::Optimizer.)doc";

static const char *__doc_popart_VariableTensor = R"doc()doc";

static const char *__doc_popart_VariableTensor_VariableTensor = R"doc()doc";

static const char *__doc_popart_VariableTensor_clone = R"doc()doc";

static const char *__doc_popart_VariableTensor_copyFromTensor = R"doc()doc";

static const char *__doc_popart_VariableTensor_getCopyFromTensor = R"doc()doc";

static const char *__doc_popart_VariableTensor_getVariableUpdateType =
    R"doc()doc";

static const char *__doc_popart_VariableTensor_setCopyFromTensor = R"doc()doc";

static const char *__doc_popart_VariableTensor_setVariableUpdateType =
    R"doc()doc";

static const char *__doc_popart_VariableTensor_variableUpdateType = R"doc()doc";

static const char *__doc_popart_VariableUpdateType = R"doc()doc";

static const char *__doc_popart_VariableUpdateType_Copy = R"doc()doc";

static const char *__doc_popart_VariableUpdateType_Gradient = R"doc()doc";

static const char *__doc_popart_VariableUpdateType_None = R"doc()doc";

static const char *__doc_popart_VirtualGraphMode =
    R"doc(Enum type used to specify a virtual graph mode.)doc";

static const char *__doc_popart_VirtualGraphMode_2 =
    R"doc(Enum type used to specify a virtual graph mode.)doc";

static const char *__doc_popart_VirtualGraphMode_Auto =
    R"doc(Use `autoVirtualGraph` transform.)doc";

static const char *__doc_popart_VirtualGraphMode_Auto_2 =
    R"doc(Use `autoVirtualGraph` transform.)doc";

static const char *__doc_popart_VirtualGraphMode_ExecutionPhases =
    R"doc(Virtual graphs are tied to execution phases.)doc";

static const char *__doc_popart_VirtualGraphMode_ExecutionPhases_2 =
    R"doc(Virtual graphs are tied to execution phases.)doc";

static const char *__doc_popart_VirtualGraphMode_Manual =
    R"doc(User must set the `virtualGraph` attribute on all ops.)doc";

static const char *__doc_popart_VirtualGraphMode_Manual_2 =
    R"doc(User must set the `virtualGraph` attribute on all ops.)doc";

static const char *__doc_popart_VirtualGraphMode_N =
    R"doc(The number of ``VirtualGraphModes`` values.)doc";

static const char *__doc_popart_VirtualGraphMode_N_2 =
    R"doc(The number of ``VirtualGraphModes`` values.)doc";

static const char *__doc_popart_VirtualGraphMode_Off =
    R"doc(Virtual graphs are not enabled.)doc";

static const char *__doc_popart_VirtualGraphMode_Off_2 =
    R"doc(Virtual graphs are not enabled.)doc";

static const char *__doc_popart_WeightDecayMode =
    R"doc(Enum type for different types of weight decay.)doc";

static const char *__doc_popart_WeightDecayMode_2 =
    R"doc(Enum type for different types of weight decay.)doc";

static const char *__doc_popart_WeightDecayMode_Decay =
    R"doc(Weight decay (e.g. AdamW))doc";

static const char *__doc_popart_WeightDecayMode_Decay_2 =
    R"doc(Weight decay (e.g. AdamW))doc";

static const char *__doc_popart_WeightDecayMode_L2Regularization =
    R"doc(L2 regularization (e.g. PyTorch-like Adam))doc";

static const char *__doc_popart_WeightDecayMode_L2Regularization_2 =
    R"doc(L2 regularization (e.g. PyTorch-like Adam))doc";

static const char *__doc_popart_anchorSumPrefix = R"doc()doc";

static const char *__doc_popart_anchorSumPrefix_2 = R"doc()doc";

static const char *__doc_popart_createRecomputedTensorId = R"doc()doc";

static const char *__doc_popart_createRecomputedTensorId_2 = R"doc()doc";

static const char *__doc_popart_cycleCountPrefix = R"doc()doc";

static const char *__doc_popart_cycleCountPrefix_2 = R"doc()doc";

static const char *__doc_popart_dotCheckFromString = R"doc()doc";

static const char *__doc_popart_dotCheckFromString_2 = R"doc()doc";

static const char *__doc_popart_error = R"doc(Exception class for popart)doc";

static const char *__doc_popart_error_2 = R"doc(Exception class for popart)doc";

static const char *__doc_popart_error_empty = R"doc()doc";

static const char *__doc_popart_error_empty_2 = R"doc()doc";

static const char *__doc_popart_error_error = R"doc()doc";

static const char *__doc_popart_error_error_2 = R"doc()doc";

static const char *__doc_popart_error_error_3 =
    R"doc(Variadic constructor for error which allows the user to use a fmt
string for the message.

throw error("This is an error reason {}", 42);)doc";

static const char *__doc_popart_error_error_4 = R"doc()doc";

static const char *__doc_popart_error_error_5 = R"doc()doc";

static const char *__doc_popart_error_error_6 = R"doc()doc";

static const char *__doc_popart_error_error_7 =
    R"doc(Variadic constructor for error which allows the user to use a fmt
string for the message.

throw error("This is an error reason {}", 42);)doc";

static const char *__doc_popart_error_error_8 = R"doc()doc";

static const char *__doc_popart_error_formatMessage =
    R"doc(As the fmt::format function can throw an exception itself we catch the
FormatError exception here and convert it to a popart exception.)doc";

static const char *__doc_popart_error_formatMessage_2 =
    R"doc(As the fmt::format function can throw an exception itself we catch the
FormatError exception here and convert it to a popart exception.)doc";

static const char *__doc_popart_error_logMessage =
    R"doc(Log the exception message)doc";

static const char *__doc_popart_error_logMessage_2 =
    R"doc(Log the exception message)doc";

static const char *__doc_popart_extractCommGroupFromAttrs =
    R"doc(Extracts CommGroup from op's attributes. If the attribute isn't set,
then the function returns a default constructed CommGroup.

Parameter ``attrs``:
    Op's attributes.

Returns:
    CommGroup that is extracted from attributes.)doc";

static const char *__doc_popart_extractCommGroupFromAttrs_2 =
    R"doc(Extracts CommGroup from op's attributes. If the attribute isn't set,
then the function returns a default constructed CommGroup.

Parameter ``attrs``:
    Op's attributes.

Returns:
    CommGroup that is extracted from attributes.)doc";

static const char *__doc_popart_extractCommGroupFromVector =
    R"doc(Extracts CommGroup from vector of two integers. If the vector is
empty, then the function returns a default constructed CommGroup.

Parameter ``vec``:
    Vector of two integers corresponding to the CommGroupType and
    replicaGroupSize.

Returns:
    CommGroup that is extracted from the input vector.)doc";

static const char *__doc_popart_extractCommGroupFromVector_2 =
    R"doc(Extracts CommGroup from vector of two integers. If the vector is
empty, then the function returns a default constructed CommGroup.

Parameter ``vec``:
    Vector of two integers corresponding to the CommGroupType and
    replicaGroupSize.

Returns:
    CommGroup that is extracted from the input vector.)doc";

static const char *__doc_popart_getDotCheckString = R"doc()doc";

static const char *__doc_popart_getDotCheckString_2 = R"doc()doc";

static const char *__doc_popart_getEdgeGradId = R"doc()doc";

static const char *__doc_popart_getEdgeGradId_2 = R"doc()doc";

static const char *__doc_popart_getErrorSource = R"doc()doc";

static const char *__doc_popart_getErrorSource_2 = R"doc()doc";

static const char *__doc_popart_getGradId = R"doc()doc";

static const char *__doc_popart_getGradId_2 = R"doc()doc";

static const char *__doc_popart_getNonGradId = R"doc()doc";

static const char *__doc_popart_getNonGradId_2 = R"doc()doc";

static const char *__doc_popart_getNonRemoteArgTensorId = R"doc()doc";

static const char *__doc_popart_getNonRemoteArgTensorId_2 = R"doc()doc";

static const char *__doc_popart_getOptMap = R"doc()doc";

static const char *__doc_popart_getOptMap_2 = R"doc()doc";

static const char *__doc_popart_getRecompId = R"doc()doc";

static const char *__doc_popart_getRecompId_2 = R"doc()doc";

static const char *__doc_popart_getRemoteArgTensorId = R"doc()doc";

static const char *__doc_popart_getRemoteArgTensorId_2 = R"doc()doc";

static const char *__doc_popart_getTensorTypeInfoMap = R"doc()doc";

static const char *__doc_popart_getUpdatedVarId = R"doc()doc";

static const char *__doc_popart_getUpdatedVarId_2 = R"doc()doc";

static const char *__doc_popart_hash_value = R"doc()doc";

static const char *__doc_popart_hash_value_2 = R"doc()doc";

static const char *__doc_popart_hash_value_3 = R"doc()doc";

static const char *__doc_popart_hash_value_4 = R"doc()doc";

static const char *__doc_popart_hash_value_5 = R"doc()doc";

static const char *__doc_popart_hash_value_6 = R"doc()doc";

static const char *__doc_popart_hostReduceGradCopyPrefix = R"doc()doc";

static const char *__doc_popart_hostReduceGradCopyPrefix_2 = R"doc()doc";

static const char *__doc_popart_hostReduceVarCopyPrefix = R"doc()doc";

static const char *__doc_popart_hostReduceVarCopyPrefix_2 = R"doc()doc";

static const char *__doc_popart_initTensorTypeInfoMap = R"doc()doc";

static const char *__doc_popart_internal_error =
    R"doc(Exception class specific to internal errors This should be used as an
assert; for states where the user should not have been able to create.)doc";

static const char *__doc_popart_internal_error_2 =
    R"doc(Exception class specific to internal errors This should be used as an
assert; for states where the user should not have been able to create.)doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter = R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_2 = R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_CorrectnessAsserter =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_CorrectnessAsserter_2 =
        R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_aFact =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_aFact_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_bps =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_bps_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkIn =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkIn_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkOut =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_checkOut_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_exe =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_exe_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getArtDivisor =
    R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_getArtDivisor_2 = R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getBaseError =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getBaseError_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getInExpected =
    R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_getInExpected_2 = R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getNElms =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getNElms_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_getOutExpected =
    R"doc()doc";

static const char *
    __doc_popart_iosizecheck_CorrectnessAsserter_getOutExpected_2 = R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_ir =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_ir_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_onnxIns =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_onnxIns_2 =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_rFact =
    R"doc()doc";

static const char *__doc_popart_iosizecheck_CorrectnessAsserter_rFact_2 =
    R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadInputSize =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadInputSize_2 =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadOutputSize =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwBadOutputSize_2 =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwIncorrectInput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwIncorrectInput_2 =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingInput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingInput_2 =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingOutput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_throwMissingOutput_2 =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_warnOfUnunsedInput =
        R"doc()doc";

static const char
    *__doc_popart_iosizecheck_CorrectnessAsserter_warnOfUnunsedInput_2 =
        R"doc()doc";

static const char *__doc_popart_iosizecheck_assertInCorrect = R"doc()doc";

static const char *__doc_popart_iosizecheck_assertInCorrect_2 = R"doc()doc";

static const char *__doc_popart_iosizecheck_assertOutCorrect = R"doc()doc";

static const char *__doc_popart_iosizecheck_assertOutCorrect_2 = R"doc()doc";

static const char *__doc_popart_isGradId = R"doc()doc";

static const char *__doc_popart_isGradId_2 = R"doc()doc";

static const char *__doc_popart_isValidTensorLocation = R"doc()doc";

static const char *__doc_popart_isValidTensorLocation_2 = R"doc()doc";

static const char *__doc_popart_memory_allocation_err = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_2 = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_clone = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_clone_2 = R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getGraphReport =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getGraphReport_2 =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getSummaryReport =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_getSummaryReport_2 =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_memory_allocation_err =
    R"doc()doc";

static const char *__doc_popart_memory_allocation_err_memory_allocation_err_2 =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_NumericsReport =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_NumericsReport_2 =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_fullReport =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_fullReport_2 =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_getRelativeErrors =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_getRelativeErrors_2 =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_relerrs = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_relerrs_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_report = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_report_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_reports = R"doc()doc";

static const char *__doc_popart_numerics_NumericsReport_reports_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_2 = R"doc()doc";

static const char
    *__doc_popart_numerics_NumericsTracker_calculateRelativeError = R"doc()doc";

static const char
    *__doc_popart_numerics_NumericsTracker_calculateRelativeError_2 =
        R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_getRelativeError =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_getRelativeError_2 =
    R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_insert = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_insert_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dA = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dA_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dAB = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dAB_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dB = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_ss_dB_2 = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_str = R"doc()doc";

static const char *__doc_popart_numerics_NumericsTracker_str_2 = R"doc()doc";

static const char *__doc_popart_operator_lshift = R"doc()doc";

static const char *__doc_popart_operator_lshift_2 = R"doc()doc";

static const char *__doc_popart_operator_lshift_3 = R"doc()doc";

static const char *__doc_popart_operator_lshift_4 =
    R"doc(Write a representation of a DeviceType to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``dt``:
    Device type reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_5 =
    R"doc(Write a representation of a DeviceConnectionType to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``dct``:
    Device connection type reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_6 =
    R"doc(Write a representation of a SyncPattern to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``sp``:
    Sync pattern reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_7 = R"doc()doc";

static const char *__doc_popart_operator_lshift_8 =
    R"doc(Write a representation of a DeviceType to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``dt``:
    Device type reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_9 =
    R"doc(Write a representation of a DeviceConnectionType to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``dct``:
    Device connection type reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_10 =
    R"doc(Write a representation of a SyncPattern to an output stream.

Parameter ``os``:
    Output stream.

Parameter ``sp``:
    Sync pattern reference.

Returns:
    The same output stream for chaining.)doc";

static const char *__doc_popart_operator_lshift_11 = R"doc()doc";

static const char *__doc_popart_operator_lshift_12 = R"doc()doc";

static const char *__doc_popart_operator_lshift_13 = R"doc()doc";

static const char *__doc_popart_operator_lshift_14 = R"doc()doc";

static const char *__doc_popart_operator_lshift_15 = R"doc()doc";

static const char *__doc_popart_operator_lshift_16 = R"doc()doc";

static const char *__doc_popart_operator_lshift_17 = R"doc()doc";

static const char *__doc_popart_operator_lshift_18 = R"doc()doc";

static const char *__doc_popart_operator_lshift_19 = R"doc()doc";

static const char *__doc_popart_operator_lshift_20 = R"doc()doc";

static const char *__doc_popart_operator_lshift_21 = R"doc()doc";

static const char *__doc_popart_operator_lshift_22 = R"doc()doc";

static const char *__doc_popart_operator_lshift_23 = R"doc()doc";

static const char *__doc_popart_operator_lshift_24 = R"doc()doc";

static const char *__doc_popart_operator_lshift_25 = R"doc()doc";

static const char *__doc_popart_operator_lshift_26 = R"doc()doc";

static const char *__doc_popart_operator_lshift_27 = R"doc()doc";

static const char *__doc_popart_operator_lshift_28 = R"doc()doc";

static const char *__doc_popart_operator_lshift_29 = R"doc()doc";

static const char *__doc_popart_operator_lshift_30 = R"doc()doc";

static const char *__doc_popart_operator_lshift_31 = R"doc()doc";

static const char *__doc_popart_operator_lshift_32 = R"doc()doc";

static const char *__doc_popart_operator_lshift_33 = R"doc()doc";

static const char *__doc_popart_operator_lshift_34 = R"doc()doc";

static const char *__doc_popart_operator_lshift_35 = R"doc()doc";

static const char *__doc_popart_operator_lshift_36 = R"doc()doc";

static const char *__doc_popart_operator_lshift_37 = R"doc()doc";

static const char *__doc_popart_operator_lshift_38 = R"doc()doc";

static const char *__doc_popart_operator_lshift_39 = R"doc()doc";

static const char *__doc_popart_operator_lshift_40 = R"doc()doc";

static const char *__doc_popart_operator_lshift_41 = R"doc()doc";

static const char *__doc_popart_operator_lshift_42 = R"doc()doc";

static const char *__doc_popart_operator_lshift_43 = R"doc()doc";

static const char *__doc_popart_optimizer_replacement_error = R"doc()doc";

static const char *__doc_popart_optimizer_replacement_error_2 = R"doc()doc";

static const char
    *__doc_popart_optimizer_replacement_error_optimizer_replacement_error =
        R"doc()doc";

static const char
    *__doc_popart_optimizer_replacement_error_optimizer_replacement_error_2 =
        R"doc()doc";

static const char
    *__doc_popart_optimizer_replacement_error_optimizer_replacement_error_3 =
        R"doc()doc";

static const char
    *__doc_popart_optimizer_replacement_error_optimizer_replacement_error_4 =
        R"doc()doc";

static const char *__doc_popart_popx_Devicex = R"doc()doc";

static const char *__doc_popart_popx_Devicex_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_4 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_Datastream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_Datastream_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_getTensorId =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_getTensorId_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_io = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_io_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_setStepIO = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_setStepIO_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_streamId = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_streamId_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_tensor = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Datastream_tensor_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Devicex = R"doc()doc";

static const char *__doc_popart_popx_Devicex_Devicex_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_InputDatastream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_InputDatastream_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_read = R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_read_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readComplete =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readComplete_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readPrefetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_InputDatastream_readPrefetch_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream = R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_OutputDatastream =
    R"doc()doc";

static const char *
    __doc_popart_popx_Devicex_OutputDatastream_OutputDatastream_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_write =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_OutputDatastream_write_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_PrefetchCallback =
    R"doc()doc";

static const char *
    __doc_popart_popx_Devicex_PrefetchCallback_PrefetchCallback_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_complete =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_complete_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_ds = R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_ds_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_fetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_fetch_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_prefetch =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_PrefetchCallback_prefetch_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostFromHostStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostFromHostStreams_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostToHostStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_anchorsHostToHostStreams_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_chBuffers = R"doc()doc";

static const char *__doc_popart_popx_Devicex_chBuffers_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRandomSeedStream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRandomSeedStream_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRngStateStream =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectRngStateStream_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectStreamToCallback =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_connectStreamToCallback_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_convCache = R"doc()doc";

static const char *__doc_popart_popx_Devicex_convCache_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyFromRemoteBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyFromRemoteBuffer_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyToRemoteBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_copyToRemoteBuffer_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCount = R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCount_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCountTensorToHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_cycleCountTensorToHost_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_d2hWeightBuffers = R"doc()doc";

static const char *__doc_popart_popx_Devicex_d2hWeightBuffers_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_deviceInfo = R"doc()doc";

static const char *__doc_popart_popx_Devicex_deviceInfo_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_doProfileChecks = R"doc()doc";

static const char *__doc_popart_popx_Devicex_doProfileChecks_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_engineIsLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_engineIsLoaded_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_executable = R"doc()doc";

static const char *__doc_popart_popx_Devicex_executable_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getAccumulationFactor =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getAccumulationFactor_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getDeviceInfo = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getDeviceInfo_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getDeviceInfo_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getDeviceInfo_4 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getEfficientlyCreatedInputTensors =
    R"doc()doc";

static const char *
    __doc_popart_popx_Devicex_getEfficientlyCreatedInputTensors_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getExecutionReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getExecutionReport_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicaOffset =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicaOffset_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicationFactor =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGlobalReplicationFactor_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGraphReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getGraphReport_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceRemoteBuffers =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceRemoteBuffers_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceStreamIds =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getHostReduceStreamIds_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getLinearlyCreatedInputTensors =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getLinearlyCreatedInputTensors_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getReplicationFactor = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getReplicationFactor_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_getRngStateToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getRngStateToHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSerializedGraph = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSerializedGraph_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSummaryReport = R"doc()doc";

static const char *__doc_popart_popx_Devicex_getSummaryReport_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_hostStreamToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_hostStreamToHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_inputStreams = R"doc()doc";

static const char *__doc_popart_popx_Devicex_inputStreams_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_ir = R"doc()doc";

static const char *__doc_popart_popx_Devicex_ir_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isEngineLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isEngineLoaded_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isReplicatedGraph = R"doc()doc";

static const char *__doc_popart_popx_Devicex_isReplicatedGraph_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_loadEngineAndConnectStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_loadEngineAndConnectStreams_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering = R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_lowering_4 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_matmulCache = R"doc()doc";

static const char *__doc_popart_popx_Devicex_matmulCache_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_nCallsToRun = R"doc()doc";

static const char *__doc_popart_popx_Devicex_nCallsToRun_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_optimizerFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_optimizerFromHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_outputStreams = R"doc()doc";

static const char *__doc_popart_popx_Devicex_outputStreams_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_pEngine = R"doc()doc";

static const char *__doc_popart_popx_Devicex_pEngine_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanConvolutions = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanConvolutions_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanMatMuls = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prePlanMatMuls_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepare = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepare_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled = R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled_3 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_prepareHasBeenCalled_4 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_readWeights = R"doc()doc";

static const char *__doc_popart_popx_Devicex_readWeights_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_reconnectInputStreams =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_reconnectInputStreams_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsFromHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsFromHost_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsToHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_remoteBufferWeightsToHost_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_rngBuffer = R"doc()doc";

static const char *__doc_popart_popx_Devicex_rngBuffer_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_run_4 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setEngineIsLoaded = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setEngineIsLoaded_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRandomSeedFromHost =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRandomSeedFromHost_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateFromHost_2 =
    R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateValue = R"doc()doc";

static const char *__doc_popart_popx_Devicex_setRngStateValue_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_stepIoSplitter = R"doc()doc";

static const char *__doc_popart_popx_Devicex_stepIoSplitter_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsFromHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsFromHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost_2 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost_3 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_weightsToHost_4 = R"doc()doc";

static const char *__doc_popart_popx_Devicex_writeWeights = R"doc()doc";

static const char *__doc_popart_popx_Devicex_writeWeights_2 = R"doc()doc";

static const char *__doc_popart_popx_Executablex = R"doc()doc";

static const char *__doc_popart_popx_Executablex_2 = R"doc()doc";

static const char *__doc_popart_popx_Executablex_3 = R"doc()doc";

static const char *__doc_popart_popx_Executablex_4 = R"doc()doc";

static const char *__doc_popart_popx_Executablex_5 = R"doc()doc";

static const char *__doc_popart_popx_Executablex_6 = R"doc()doc";

static const char *__doc_popart_popx_IrLowering = R"doc()doc";

static const char *__doc_popart_popx_IrLowering_2 = R"doc()doc";

static const char *__doc_popart_popx_IrLowering_3 = R"doc()doc";

static const char *__doc_popart_popx_IrLowering_4 = R"doc()doc";

static const char *__doc_popart_popx_popType = R"doc()doc";

static const char *__doc_popart_popx_popType_2 = R"doc()doc";

static const char *__doc_popart_popx_popType_3 = R"doc()doc";

static const char *__doc_popart_popx_popType_4 = R"doc()doc";

static const char *__doc_popart_reservedAccl1Prefix = R"doc()doc";

static const char *__doc_popart_reservedAccl1Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAccl2Prefix = R"doc()doc";

static const char *__doc_popart_reservedAccl2Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAccl3Prefix = R"doc()doc";

static const char *__doc_popart_reservedAccl3Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAcclFinalOutPrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclFinalOutPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAcclPrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAcclToReducePrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclToReducePrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAcclToUpdatePrefix = R"doc()doc";

static const char *__doc_popart_reservedAcclToUpdatePrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAccumPrefix = R"doc()doc";

static const char *__doc_popart_reservedAccumPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAccumulatorPrefixes = R"doc()doc";

static const char *__doc_popart_reservedAccumulatorPrefixes_2 = R"doc()doc";

static const char *__doc_popart_reservedAdamUpdaterPrefix = R"doc()doc";

static const char *__doc_popart_reservedAdamUpdaterPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedAdaptiveUpdaterPrefix = R"doc()doc";

static const char *__doc_popart_reservedAdaptiveUpdaterPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedConcatInitPrefix = R"doc()doc";

static const char *__doc_popart_reservedConcatInitPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedConstValuePrefix = R"doc()doc";

static const char *__doc_popart_reservedConstValuePrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta1Prefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta1Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta2Prefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamBeta2Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamEpsPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdamGradientScalingPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveAlphaPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveAlphaPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveEpsPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveGradientScalingPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveMomentumPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultAdaptiveMomentumPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultDampeningScaleFactor1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultDampeningScaleFactor1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultLearningRatePrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultLearningRatePrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultLossScalingPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultMaxWeightNormPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultMaxWeightNormPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate0Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledLearningRate1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledMomentum1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledMomentum1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledWeightDecay1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultScaledWeightDecay1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultStepPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayPrefix = R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayScaleFactor0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedDefaultWeightDecayScaleFactor0Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedGradientPrefix = R"doc()doc";

static const char *__doc_popart_reservedGradientPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedIndexPrefix = R"doc()doc";

static const char *__doc_popart_reservedIndexPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedInitPrefix = R"doc()doc";

static const char *__doc_popart_reservedInitPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedLambR1SqPrefix = R"doc()doc";

static const char *__doc_popart_reservedLambR1SqPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedLambR2SqPrefix = R"doc()doc";

static const char *__doc_popart_reservedLambR2SqPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedLoopCondPrefix = R"doc()doc";

static const char *__doc_popart_reservedLoopCondPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedLoopIteratorPrefix = R"doc()doc";

static const char *__doc_popart_reservedLoopIteratorPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedLossScalingPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedOptimizerPrefixes = R"doc()doc";

static const char *__doc_popart_reservedOptimizerPrefixes_2 = R"doc()doc";

static const char *__doc_popart_reservedOptimizerStatePrefixes = R"doc()doc";

static const char *__doc_popart_reservedOptimizerStatePrefixes_2 = R"doc()doc";

static const char *__doc_popart_reservedPrefixes = R"doc()doc";

static const char *__doc_popart_reservedPrefixes_2 = R"doc()doc";

static const char *__doc_popart_reservedRandomSeedPrefix = R"doc()doc";

static const char *__doc_popart_reservedRandomSeedPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedRemoteArgPrefix = R"doc()doc";

static const char *__doc_popart_reservedRemoteArgPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedRestoredPrefix = R"doc()doc";

static const char *__doc_popart_reservedRestoredPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta1Prefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta1Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta2Prefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamBeta2Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamEpsPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamGradientScalingPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdamGradientScalingPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveAlphaPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveAlphaPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveEpsPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveEpsPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveGradientScalingPrefix =
    R"doc()doc";

static const char
    *__doc_popart_reservedSpecificAdaptiveGradientScalingPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveMomentumPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificAdaptiveMomentumPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificDampeningScaleFactor1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificDampeningScaleFactor1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificLearningRatePrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificLearningRatePrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificLossScalingPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificLossScalingPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificMaxWeightNormPrefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificMaxWeightNormPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate0Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate0Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledLearningRate1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledMomentum1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledMomentum1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledWeightDecay1Prefix =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificScaledWeightDecay1Prefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificStepPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedSpecificWeightDecayPrefix = R"doc()doc";

static const char *__doc_popart_reservedSpecificWeightDecayPrefix_2 =
    R"doc()doc";

static const char *__doc_popart_reservedSpecificWeightDecayScaleFactor0Prefix =
    R"doc()doc";

static const char
    *__doc_popart_reservedSpecificWeightDecayScaleFactor0Prefix_2 = R"doc()doc";

static const char *__doc_popart_reservedStashedPrefix = R"doc()doc";

static const char *__doc_popart_reservedStashedPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedStepPrefix = R"doc()doc";

static const char *__doc_popart_reservedStepPrefix_2 = R"doc()doc";

static const char *__doc_popart_reservedUpdatedVarPrefix = R"doc()doc";

static const char *__doc_popart_reservedUpdatedVarPrefix_2 = R"doc()doc";

static const char *__doc_popart_stripAllReservedPrefixes = R"doc()doc";

static const char *__doc_popart_stripAllReservedPrefixes_2 = R"doc()doc";

static const char *__doc_popart_syncPatternFromString = R"doc()doc";

static const char *__doc_popart_syncPatternFromString_2 = R"doc()doc";

static const char *__doc_popart_syncPatternToString = R"doc()doc";

static const char *__doc_popart_syncPatternToString_2 = R"doc()doc";

static const char *__doc_popart_toGCLCommGroup =
    R"doc(Converts give CommGroup to GCL's CommGroup type.

Parameter ``input``:
    PopART CommGroup.

Returns:
    GCL CommGroup.)doc";

static const char *__doc_popart_toGCLCommGroup_2 =
    R"doc(Converts give CommGroup to GCL's CommGroup type.

Parameter ``input``:
    PopART CommGroup.

Returns:
    GCL CommGroup.)doc";

static const char *__doc_popart_toString = R"doc()doc";

static const char *__doc_popart_toString_2 = R"doc()doc";

static const char *__doc_popart_toString_3 = R"doc()doc";

static const char *__doc_popart_toString_4 = R"doc()doc";

static const char *__doc_poplar_OptionFlags = R"doc()doc";

static const char *__doc_poplar_OptionFlags_2 = R"doc()doc";

static const char *__doc_poplar_Target = R"doc()doc";

static const char *__doc_poplar_Target_2 = R"doc()doc";

static const char *__doc_poprithms_logging_TimePartitionLogger = R"doc()doc";

static const char *__doc_poprithms_logging_TimePartitionLogger_2 = R"doc()doc";

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

static const char *__doc_std_hash_11 = R"doc()doc";

static const char *__doc_std_hash_12 = R"doc()doc";

static const char *__doc_std_hash_13 = R"doc()doc";

static const char *__doc_std_hash_14 = R"doc()doc";

static const char *__doc_std_hash_15 = R"doc()doc";

static const char *__doc_std_hash_16 = R"doc()doc";

static const char *__doc_std_hash_17 = R"doc()doc";

static const char *__doc_std_hash_18 = R"doc()doc";

static const char *__doc_std_hash_19 = R"doc()doc";

static const char *__doc_std_hash_20 = R"doc()doc";

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

static const char *__doc_std_hash_operator_call_11 = R"doc()doc";

static const char *__doc_std_hash_operator_call_12 = R"doc()doc";

static const char *__doc_std_hash_operator_call_13 = R"doc()doc";

static const char *__doc_std_hash_operator_call_14 = R"doc()doc";

static const char *__doc_std_hash_operator_call_15 = R"doc()doc";

static const char *__doc_std_hash_operator_call_16 = R"doc()doc";

static const char *__doc_std_hash_operator_call_17 = R"doc()doc";

static const char *__doc_std_hash_operator_call_18 = R"doc()doc";

static const char *__doc_std_hash_operator_call_19 = R"doc()doc";

static const char *__doc_std_hash_operator_call_20 = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
