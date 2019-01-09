#include <functional>

#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

/*
 * This registers ONNX operators and shape inference for the Graphcore custom
 * operators.
 */

namespace ONNX_NAMESPACE {

void SubsampleShapeInference(InferenceContext &ctx);

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
      fail_shape_inference("Attribute strides has incorrect size");
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
      fail_shape_inference("Attribute stride of zero");
    }
  }
}

extern size_t dbg_count_check_Subsample_AiGraphcore_ver1;

static const char subsampleDoc[] =
    "Subsample takes every Nth element of a tensor.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    Subsample,
    AiGraphcore,
    poponnx::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(subsampleDoc)
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

static bool registerOps() {
  auto &d = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion(poponnx::Domain::ai_graphcore, 1, 1);

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Subsample)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
