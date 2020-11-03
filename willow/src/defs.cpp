// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <functional>

#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op/receptive.hpp>
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
void DetachShapeInference(InferenceContext &ctx);
void CallShapeInference(InferenceContext &ctx);
void DynamicUpdateShapeInference(InferenceContext &ctx);
void DynamicSliceInference(InferenceContext &ctx);
void DynamicZeroShapeInference(InferenceContext &ctx);
void DynamicAddShapeInference(InferenceContext &ctx);
void MultiConvShapeInference(InferenceContext &ctx);
void NopShapeInference(InferenceContext &ctx);
void ShapedDropoutShapeInference(InferenceContext &ctx);
void Expm1ShapeInference(InferenceContext &ctx);
void Log1pShapeInference(InferenceContext &ctx);

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
  const unsigned int X_IN = 0;

  // Unused: const unsigned int Y_OUT    = 0;
  const unsigned int MEAN_OUT = 1;
  const unsigned int STD_OUT  = 2;

  const unsigned int NUM_OUTPUTS = 3;
  for (unsigned int i = 0; i < NUM_OUTPUTS; i++) {
    propagateElemTypeFromInputToOutput(ctx, X_IN, i);
  }

  propagateShapeFromInputToOutput(ctx, X_IN, 0);

  int32_t num_groups = 1;
  getAttribute(ctx, "num_groups", num_groups);

  auto input_shape = ctx.getInputType(X_IN)->tensor_type().shape();

  if (input_shape.dim_size() > 0 && input_shape.dim(0).has_dim_value()) {
    auto *output_shape_mean =
        ctx.getOutputType(MEAN_OUT)->mutable_tensor_type()->mutable_shape();
    output_shape_mean->add_dim()->set_dim_value(num_groups *
                                                input_shape.dim(0).dim_value());

    auto *output_shape_stdev =
        ctx.getOutputType(STD_OUT)->mutable_tensor_type()->mutable_shape();
    output_shape_stdev->add_dim()->set_dim_value(
        num_groups * input_shape.dim(0).dim_value());
  }
}

void PrintTensorShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ScaleShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void LSTMShapeInference(InferenceContext &ctx) {
  const unsigned int X_IN       = 0;
  const unsigned int WEIGHTS_IN = 1;

  const unsigned int OUTPUT_OUT     = 0;
  const unsigned int CELL_STATE_OUT = 1;

  propagateElemTypeFromInputToOutput(ctx, X_IN, OUTPUT_OUT);
  propagateElemTypeFromInputToOutput(ctx, X_IN, CELL_STATE_OUT);

  // we need the first 2 input shapes for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape   = ctx.getInputType(X_IN)->tensor_type().shape();
  auto weights_shape = ctx.getInputType(WEIGHTS_IN)->tensor_type().shape();

  auto seq_length = input_shape.dim(0);
  auto batch_size = input_shape.dim(1);
  auto input_size = input_shape.dim(2);

  auto hidden_size = weights_shape.dim(2);

  int64_t output_full_sequence = 1;
  getAttribute(ctx, "output_full_sequence", output_full_sequence);

  auto *output_shape =
      ctx.getOutputType(OUTPUT_OUT)->mutable_tensor_type()->mutable_shape();
  if (output_full_sequence != 0) {
    *output_shape->add_dim() = seq_length;
  }
  *output_shape->add_dim() = batch_size;
  *output_shape->add_dim() = hidden_size;

  auto *cell_state_shape =
      ctx.getOutputType(CELL_STATE_OUT)->mutable_tensor_type()->mutable_shape();
  *cell_state_shape->add_dim() = batch_size;
  *cell_state_shape->add_dim() = hidden_size;
}

void GeluShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DetachShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void CallShapeInference(InferenceContext &ctx) {
  auto num_inputs = ctx.getNumInputs();

  std::vector<const ONNX_NAMESPACE::TypeProto *> subgraph_input_types;

  // Run inferencing on the subgraph
  std::vector<const ONNX_NAMESPACE::TypeProto *> subgraph_output_types;
  GraphInferencer *graphInferencer = ctx.getGraphAttributeInferencer("callee");
  if (graphInferencer) {
    std::vector<const ONNX_NAMESPACE::TensorProto *> input_data;
    for (size_t i = 0; i < num_inputs; ++i) {
      input_data.push_back(ctx.getInputData(i));
      subgraph_input_types.push_back(ctx.getInputType(i));
    }

    subgraph_output_types =
        graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    if (subgraph_output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
      auto *subgraph_output_type = subgraph_output_types[i];
      auto *output_type          = ctx.getOutputType(i);

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Graph 'callee' subgraph outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, output_type);

      // merge shape; type already propagated
      auto &subgraph_output_tensor_type = subgraph_output_type->tensor_type();
      auto *mutable_call_output_tensor_type =
          output_type->mutable_tensor_type();

      mergeInShapeInfo(subgraph_output_tensor_type,
                       *mutable_call_output_tensor_type);
    }
  }
}

void DynamicUpdateShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicSliceShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicZeroShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicAddShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

template <unsigned int infer_shape_index>
void LossShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  std::string reduction = getAttribute(ctx, "reduction", "Mean");

  if (reduction == "None") {
    if (hasInputShape(ctx, infer_shape_index)) {
      propagateShapeFromInputToOutput(ctx, infer_shape_index, 0);
    }
  } else {
    if (reduction != "Mean" && reduction != "Sum") {
      throw popart::internal_error("No loss reduction type for {}", reduction);
    }

    // Scalar output
    updateOutputShape(ctx, 0, TensorShapeProto());
  }
}

void MultiConvShapeInference(InferenceContext &ctx) {
  auto num_outputs              = ctx.getNumOutputs();
  int64_t cumulativeSpatialDims = 0;

  popart::Shape flatPads;
  popart::Shape flatStrides;
  popart::Shape flatDilations;
  getRepeatedAttribute(ctx, "pads", flatPads);
  getRepeatedAttribute(ctx, "strides", flatStrides);
  getRepeatedAttribute(ctx, "dilations", flatDilations);

  for (size_t outIdx = 0; outIdx < num_outputs; ++outIdx) {
    size_t dataIdx    = (2 * outIdx);
    size_t weightsIdx = (2 * outIdx) + 1;

    propagateElemTypeFromInputToOutput(ctx, dataIdx, outIdx);

    auto dataShape    = ctx.getInputType(dataIdx)->tensor_type().shape();
    auto weightsShape = ctx.getInputType(weightsIdx)->tensor_type().shape();

    auto *outputShape =
        ctx.getOutputType(outIdx)->mutable_tensor_type()->mutable_shape();

    *outputShape->add_dim() = dataShape.dim(0);    // batch size
    *outputShape->add_dim() = weightsShape.dim(0); // num output channels

    auto dataSize    = dataShape.dim_size();
    auto spatialSize = dataSize - 2;

    // Unflatten parameters
    popart::Shape pads;
    popart::Shape strides;
    popart::Shape dilations;
    if (flatPads.empty()) {
      pads.assign(spatialSize * 2, 0);
    } else {
      const auto cumulativePads = cumulativeSpatialDims * 2;
      pads                      = {flatPads.begin() + cumulativePads,
              flatPads.begin() + cumulativePads + (spatialSize * 2)};
    }
    if (flatStrides.empty()) {
      strides.assign(spatialSize, 1);
    } else {
      strides = {flatStrides.begin() + cumulativeSpatialDims,
                 flatStrides.begin() + cumulativeSpatialDims + spatialSize};
    }
    if (flatDilations.empty()) {
      dilations.assign(spatialSize, 1);
    } else {
      dilations = {flatDilations.begin() + cumulativeSpatialDims,
                   flatDilations.begin() + cumulativeSpatialDims + spatialSize};
    }
    cumulativeSpatialDims += spatialSize;

    popart::Shape spatialDShape;
    popart::Shape spatialKShape;
    for (int i = 2; i < dataSize; i++) {
      spatialDShape.push_back(dataShape.dim(i).dim_value());
      spatialKShape.push_back(weightsShape.dim(i).dim_value());
    }

    popart::Shape spatialOutShape =
        popart::HasReceptiveFieldOp::getSpatialOutShape(
            spatialDShape,
            spatialKShape,
            pads,
            strides,
            dilations,
            popart::AutoPad::NOTSET);

    for (auto dim : spatialOutShape) {
      outputShape->add_dim()->set_dim_value(dim);
    }
  }
}

void NopShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ShapedDropoutShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void Atan2ShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (!hasNInputShapes(ctx, 2)) {
    fail_type_inference("Atan2 requires two inputtensors with shapes.");
  }

  bidirectionalBroadcastShapeInference(
      ctx.getInputType(0)->tensor_type().shape(),
      ctx.getInputType(1)->tensor_type().shape(),
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void Expm1ShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void Log1pShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

extern size_t dbg_count_check_GroupNormalization_AiGraphcore_ver1;
extern size_t dbg_count_check_Subsample_AiGraphcore_ver1;
extern size_t dbg_count_check_PrintTensor_AiGraphcore_ver1;
extern size_t dbg_count_check_Scale_AiGraphcore_ver1;
extern size_t dbg_count_check_LSTM_AiGraphcore_ver1;
extern size_t dbg_count_check_Gelu_AiGraphcore_ver1;
extern size_t dbg_count_check_Detach_AiGraphcore_ver1;
extern size_t dbg_count_check_Call_AiGraphcore_ver1;
extern size_t dbg_count_check_L1_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicUpdate_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicSlice_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicZero_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicAdd_AiGraphcore_ver1;
extern size_t dbg_count_check_MultiConv_AiGraphcore_ver1;
extern size_t dbg_count_check_Nop_AiGraphcore_ver1;
extern size_t dbg_count_check_ShapedDropout_AiGraphcore_ver1;
extern size_t dbg_count_check_Atan2_AiGraphcore_ver1;
extern size_t dbg_count_check_Expm1_AiGraphcore_ver1;
extern size_t dbg_count_check_Log1p_AiGraphcore_ver1;

static const char groupnormalizationDoc[] =
    "GroupNormalization applies Group Normalization over a mini-batch of "
    "input";

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
        .TypeAndShapeInferenceFunction(GroupNormalizationShapeInference))

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
        .TypeAndShapeInferenceFunction(SubsampleShapeInference))

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
        .Attr("title",
              "A sting to use for the title of the tensor. Will use tensor "
              "name if not provided.",
              AttributeProto::STRING,
              "")
        .TypeAndShapeInferenceFunction(PrintTensorShapeInference))

static const char scaleDoc[] =
    "Scale takes one input data (Tensor<float>) and produces one output data "
    "(Tensor<float>) whose value is the input data tensor scaled "
    "element-wise.";

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
        .TypeAndShapeInferenceFunction(ScaleShapeInference))

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
        .Input(3, "InitState", "The initial state", "T")
        .Output(0, "Output", "Output tensor", "T")
        .Output(1, "CellState", "The lstm cell state", "T")
        // Optional (training) output 2 ignored
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .Attr("output_full_sequence",
              "If true, the lstm returns the entire sequence of outputs, "
              "otherwise it just returns the final output.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(LSTMShapeInference))

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
        .TypeAndShapeInferenceFunction(GeluShapeInference))

static const char detachDoc[] =
    "An IdentityOp that doesn't return any grad ops. This allows you to "
    "disconnect the flow of gradients when creating the backwards pass";

ONNX_OPERATOR_SET_SCHEMA_EX(
    Detach,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(detachDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)",
                         "tensor(int32)",
                         "tensor(float16)",
                         "tensor(bool)"},
                        "Do not constrain tensors")
        .TypeAndShapeInferenceFunction(DetachShapeInference))

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
        .TypeAndShapeInferenceFunction(CallShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    DynamicUpdate,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Update a subtensor at offsets read from a tensor.")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
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
        .Attr("axes", "Axes along which to update.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .Attr("noOverlap", ".", AttributeProto::INT, true)
        .TypeAndShapeInferenceFunction(DynamicUpdateShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    DynamicSlice,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Slice a tensor based on offsets specified by a tensor.")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "index", "Index tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
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
        .Attr("axes", "Axes along which to update.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .Attr("noOverlap", ".", AttributeProto::INT, true)
        .TypeAndShapeInferenceFunction(DynamicSliceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    DynamicZero,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Creates a copy of 'tensor' with a slice at 'offset' set to "
                "zero, e.g. out = tensor, out[offset] = 0.0")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("axes", "Axes along which to erase.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(DynamicZeroShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    DynamicAdd,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Creates a copy of 'tensor' with 'slice' added at 'offset', "
                "e.g. out = tensor, out[offset] += slice")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("axes", "Axes along which to add.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(DynamicAddShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    L1,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Calculates the absolute values of each element in the input "
                "and optionally mean/sum reduce the output")
        .Input(
            0,
            "input",
            "Any shape tensor for which to calculate to calculate the scaled "
            "absolute values or l1 norm",
            "T")
        .Output(0, "output", "The scaled absolute values / l1 norm", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(uint32)",
             "tensor(int32)"},
            "Constrain input and output types to float(32/16) and (u)int32 "
            "tensors.")
        .Attr("lambda",
              "Scaling factor for the loss",
              AttributeProto::FLOAT,
              true)
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING)
        .TypeAndShapeInferenceFunction(LossShapeInference<0>))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Nll,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Calculates the negative likelihood loss on based on the "
                "probability and label inputs")
        .Input(0, "probs", "Tensor of shape [D1, ..., DN, NumClasses]", "T")
        .Input(1,
               "label",
               "Tensor of shape [D1, ..., DN] where each element is a class "
               "index",
               "Tind")
        .Output(0,
                "output",
                "The negative log likelihood loss, possibly "
                "sum/mean reduced",
                "T")
        .TypeConstraint("T",
                        {"tensor(float16)", "tensor(float)"},
                        "Floating Point Tensors")
        .TypeConstraint("Tind",
                        {"tensor(int16)", "tensor(int32)"},
                        "Integer types")
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING,
              true)
        .Attr("ignoreIndex",
              "If non negative, ignores targets with the same index so that "
              "they do not contribute to the loss",
              AttributeProto::INT,
              false)
        .Attr("inputIsLogProbability",
              "Specifies if the input tensor contains log-probabilities or raw "
              "probabilities",
              AttributeProto::INT,
              false)
        .TypeAndShapeInferenceFunction(LossShapeInference<1>))

ONNX_OPERATOR_SET_SCHEMA_EX(
    IdentityLoss,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Outputs the identity of the inputs, or the sum or mean "
                "of the inputs: generally used as a utility to provide the "
                "reduction of other losses")
        .Input(0, "input", "Tensor inputs to the loss", "T")
        .Output(0,
                "outputs",
                "A Tensor (identical to input) or scalar output depending on "
                "whether reduction is specified",
                "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING,
              true)
        .TypeAndShapeInferenceFunction(LossShapeInference<0>))

ONNX_OPERATOR_SET_SCHEMA_EX(
    MultiConv,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("MultiConv comprises multiple convolution operations that can "
                "be run in parallel")
        .Input(0,
               "inputs",
               "List of inputs to the convolutions, in pairwise 'data, "
               "weights' order",
               "T",
               OpSchema::Variadic)
        .Output(0,
                "outputs",
                "List of outputs from the convolutions",
                "T",
                OpSchema::Variadic)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("strides",
              "The concatenated strides for each convoltion.",
              AttributeProto::INTS,
              true)
        .Attr("pads",
              "The concatenated pads for each convoltion.",
              AttributeProto::INTS,
              true)
        .Attr("dilations",
              "The concatenated dilations for each convoltion.",
              AttributeProto::INTS,
              true)
        .Attr("planType", "The plan type.", AttributeProto::STRING, true)
        .Attr("perConvReservedTiles",
              "The number of tiles reserved per convolution.",
              AttributeProto::INT,
              true)
        .Attr("perConvReservedTiles",
              "The number of tiles reserved per convolution.",
              AttributeProto::INT,
              true)
        .Attr("cycleBackOff",
              "The cycle back-off proportion.",
              AttributeProto::FLOAT,
              true)
        .Attr(popart::sAvailMemAttribute,
              "The available memory proportion per convolution.",
              AttributeProto::FLOATS,
              true)
        .Attr(popart::sPartialsTypeAttribute,
              "The partials type when computing each convolution.",
              AttributeProto::FLOATS,
              true)
        .Attr("numConvs",
              "The number convolutions that the MultiConv comprises.",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(MultiConvShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Nop,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("No operation, for debugging purposes.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(NopShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    ShapedDropout,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies a shaped dropout to the input tensor.")
        .Input(0, "input", "Tensor to apply shaped dropout to", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("shape",
              "Shape of dropout mask. Must be broadcastable to the input",
              AttributeProto::INTS,
              true)
        .Attr("ratio",
              "Probability of dropping an input feature (default = 0.5)",
              AttributeProto::FLOAT,
              0.5f)
        .TypeAndShapeInferenceFunction(ShapedDropoutShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Atan2,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Atan2(Y, X)")
        .Input(0, "Y", "First input tensor", "T")
        .Input(1, "X", "Second input tensor", "T")
        .Output(0, "Theta", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Atan2ShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Expm1,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Exp(x) - 1.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Expm1ShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Log1p,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Log(x + 1).")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Log1pShapeInference))

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
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Detach)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DynamicUpdate)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Call)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, L1)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Nll)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, IdentityLoss)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, MultiConv)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Nop)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, ShapedDropout)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Atan2)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
