#include <functional>

#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

/*
 * This registers ONNX operators and shape inference for the Graphcore custom
 * operators.
 */

namespace ONNX_NAMESPACE {

void SubsampleShapeInference(InferenceContext &ctx);
void GroupNormalizationShapeInference(InferenceContext &ctx);
void PrintTensorShapeInference(InferenceContext &ctx);
void ScaleShapeInference(InferenceContext &ctx);
void LSTMShapeInference(InferenceContext &ctx);
void GeluShapeInference(InferenceContext &ctx);
void CallShapeInference(InferenceContext &ctx);

void SubsampleShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need the first input shape for this inference.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size());

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size")
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  auto *output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int d = 0; d < n_input_dims; d++) {
    if (strides[d] != 0) {
      *output_shape->add_dim() = (input_shape.dim(d) / strides[d]);
    } else {
      fail_shape_inference("Attribute stride of zero")
    }
  }
}

// Do the same as BatchNormalization & InstanceNormalization in onnx
void GroupNormalizationShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void PrintTensorShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ScaleShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void LSTMShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need the first 2 input shapes for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape   = ctx.getInputType(0)->tensor_type().shape();
  auto weights_shape = ctx.getInputType(0)->tensor_type().shape();

  auto seq_length  = input_shape.dim(0);
  auto batch_size  = input_shape.dim(1);
  auto input_size  = input_shape.dim(2);
  auto hidden_size = weights_shape.dim(2);

  int64_t output_full_sequence = 1;
  getAttribute(ctx, "output_full_sequence", output_full_sequence);

  auto *output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  if (output_full_sequence != 0) {
    *output_shape->add_dim() = seq_length;
  }
  *output_shape->add_dim() = batch_size;
  *output_shape->add_dim() = hidden_size;

  auto *cell_state_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *cell_state_shape->add_dim() = batch_size;
  *cell_state_shape->add_dim() = hidden_size;
}

void GeluShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void CallShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

extern size_t dbg_count_check_GroupNormalization_AiGraphcore_ver1;
extern size_t dbg_count_check_Subsample_AiGraphcore_ver1;
extern size_t dbg_count_check_PrintTensor_AiGraphcore_ver1;
extern size_t dbg_count_check_Scale_AiGraphcore_ver1;
extern size_t dbg_count_check_LSTM_AiGraphcore_ver1;
extern size_t dbg_count_check_Gelu_AiGraphcore_ver1;
extern size_t dbg_count_check_Call_AiGraphcore_ver1;

static const char groupnormalizationDoc[] =
    "GroupNormalization applies Group Normalization over a mini-batch of input";

ONNX_OPERATOR_SET_SCHEMA_EX(
    GroupNormalization,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(groupnormalizationDoc)
        .Input(0, "X", "Input tensor", "T")
        .Input(1,
               "Scale",
               "The input 1-dimensional scale tensor of size C.",
               "T")
        .Input(2, "Bias", "The input 1-dimensional bias tensor of size C.", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Output(1, "Mean", "The mean after GroupNormalization operator", "T")
        .Output(2, "Var", "The variance after GroupNormalization operator", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("num_groups", "The number of groups", AttributeProto::INT, false)
        .Attr("epsilon",
              "The epsilon value to use to avoid division by zero.",
              AttributeProto::FLOAT,
              1e-5f)
        .TypeAndShapeInferenceFunction(GroupNormalizationShapeInference));

ONNX_OPERATOR_SET_SCHEMA_EX(
    Subsample,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Subsample takes every Nth element of a tensor.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("strides",
              "Strides in each of the dimensions.",
              AttributeProto::INTS,
              false)
        .TypeAndShapeInferenceFunction(SubsampleShapeInference));

ONNX_OPERATOR_SET_SCHEMA_EX(
    PrintTensor,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("PrintTensor prints the value of a tensor.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("print_gradient",
              "Should the gradient tensor also be printed.",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(PrintTensorShapeInference));

static const char scaleDoc[] =
    "Scale takes one input data (Tensor<float>) and produces one output data "
    "(Tensor<float>) whose value is the input data tensor scaled element-wise.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    Scale,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(scaleDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("scale", "The scale to apply", AttributeProto::FLOAT, true)
        .TypeAndShapeInferenceFunction(ScaleShapeInference));

ONNX_OPERATOR_SET_SCHEMA_EX(
    LSTM,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("")
        .Input(0, "X", "The input tensor", "T")
        .Input(1, "Weights", "The concatenated input and output weights", "T")
        .Input(2, "Bias", "The biases", "T")
        .Input(0, "InitState", "The initial state", "T")
        .Output(0, "Output", "Output tensor", "T")
        .Output(1, "CellState", "The lstm cell state", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .Attr("output_full_sequence",
              "If true, the lstm returns the entire sequence of outputs, "
              "otherwise it just returns the final output.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(LSTMShapeInference));

ONNX_OPERATOR_SET_SCHEMA_EX(
    Gelu,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies the Gaussian Error Linear Units function.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(GeluShapeInference));

ONNX_OPERATOR_SET_SCHEMA_EX(
    Call,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Call instantiates a call to a subgraph.")
        .Input(0,
               "inputs",
               "List of inputs to the subgraph",
               "T",
               OpSchema::Variadic)
        .Output(0,
                "outputs",
                "List of outputs from the subgraph",
                "T",
                OpSchema::Variadic)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("callee",
              "The subgraph to call into.",
              AttributeProto::GRAPH,
              true)
        .TypeAndShapeInferenceFunction(CallShapeInference));

static bool registerOps() {
  auto &d = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion(popart::Domain::ai_graphcore, 1, 1);

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, GroupNormalization)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Subsample)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, PrintTensor)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Scale)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, LSTM)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Gelu)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Call)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
