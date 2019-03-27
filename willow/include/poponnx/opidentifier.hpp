#ifndef GUARD_NEURALNET_OPIDENTIFIER_HPP
#define GUARD_NEURALNET_OPIDENTIFIER_HPP

#include <poponnx/attributes.hpp>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>

namespace poponnx {

namespace Domain {
const static char *ai_onnx      = "ai.onnx";
const static char *ai_onnx_ml   = "ai.onnx.ml";
const static char *ai_graphcore = "ai.graphcore";
} // namespace Domain

// Default opset versions for domains
const static int64_t defaultAiOnnxOpset      = 9;
const static int64_t defaultAiOnnxMlOpset    = 1;
const static int64_t defaultAiGraphcoreOpset = 1;

struct NumInputs {

  int min;
  int max;

  NumInputs() : min(0), max(0) {}
  NumInputs(int f) : min(f), max(f) {}
  NumInputs(int _min, int _max) : min(_min), max(_max) {}
};

// The Op identifier is defined by ONNX a tuple
// (https://github.com/onnx/onnx/blob/master/docs/Versioning.md)
// domain.type:version
struct OperatorIdentifier {
  OperatorIdentifier(const OpDomain &_domain,
                     const OpType &_type,
                     OpVersion _version,
                     NumInputs inputs = {},
                     int outputs      = 0)
      : domain(_domain), type(_type), version(_version), numInputs(inputs),
        numOutputs(outputs) {

    // If no domain specified assume it is the default
    if (domain == "") {
      domain = Domain::ai_onnx;
    }
  }

  OpDomain domain;
  OpType type;
  OpVersion version;

  NumInputs numInputs;
  int numOutputs;

  bool operator==(const OperatorIdentifier &rhs) const {
    return (domain == rhs.domain && type == rhs.type && version == rhs.version);
  }

  bool operator!=(const OperatorIdentifier &rhs) const {
    return !(*this == rhs);
  }
};

// The following does not work as we are in the poponnx namesapace >>
// template<>  struct less<OperatorIdentifier>
struct OperatorIdentifierLess {
  bool operator()(const OperatorIdentifier &lhs,
                  const OperatorIdentifier &rhs) const {
    if (lhs.domain < rhs.domain) {
      return true;
    } else if (lhs.domain > rhs.domain) {
      return false;
    } else {
      if (lhs.type < rhs.type) {
        return true;
      } else if (lhs.type > rhs.type) {
        return false;
      } else {
        if (lhs.version < rhs.version) {
          return true;
        } else {
          return false;
        }
      }
    }
  }
};

struct AiOnnxOperatorIdentifierV8 : public OperatorIdentifier {
  AiOnnxOperatorIdentifierV8(const OpType &_type)
      : OperatorIdentifier(Domain::ai_onnx, _type, 8) {}
};

struct AiOnnxOperatorIdentifierV9 : public OperatorIdentifier {
  AiOnnxOperatorIdentifierV9(const OpType &_type,
                             NumInputs inputs = {},
                             int outputs      = 0)
      : OperatorIdentifier(Domain::ai_onnx, _type, 9, inputs, outputs) {}
};

struct AiGraphcoreOpIdV1 : public OperatorIdentifier {
  AiGraphcoreOpIdV1(const OpType &_type, NumInputs inputs = {}, int outputs = 0)
      : OperatorIdentifier(Domain::ai_graphcore, _type, 1, inputs, outputs) {}
};

std::ostream &operator<<(std::ostream &os, const OperatorIdentifier &opid);

// TODO :  Generating this list of operators from the md files
namespace Onnx {

namespace Operators {

const static OperatorIdentifier
    ConstantLike_9(Domain::ai_onnx, "ConstantLike", 1, 1, 1);

const static OperatorIdentifier Abs_1(Domain::ai_onnx, "Abs", 1, 1, 1);
const static OperatorIdentifier Abs_6(Domain::ai_onnx, "Abs", 6, 1, 1);
const static OperatorIdentifier Acos_7(Domain::ai_onnx, "Acos", 7, 1, 1);
const static OperatorIdentifier Acosh_9(Domain::ai_onnx, "Acosh", 9, 1, 1);
const static OperatorIdentifier Add_1(Domain::ai_onnx, "Add", 1, 2, 1);
const static OperatorIdentifier Add_6(Domain::ai_onnx, "Add", 6, 2, 1);
const static OperatorIdentifier Add_7(Domain::ai_onnx, "Add", 7, 2, 1);
const static OperatorIdentifier And_1(Domain::ai_onnx, "And", 1, 2, 1);
const static OperatorIdentifier And_7(Domain::ai_onnx, "And", 7, 2, 1);
const static OperatorIdentifier ArgMax_1(Domain::ai_onnx, "ArgMax", 1, 1, 1);
const static OperatorIdentifier ArgMin_1(Domain::ai_onnx, "ArgMin", 1, 1, 1);
const static OperatorIdentifier Asin_7(Domain::ai_onnx, "Asin", 7, 1, 1);
const static OperatorIdentifier Asinh_9(Domain::ai_onnx, "Asinh", 9, 1, 1);
const static OperatorIdentifier Atan_7(Domain::ai_onnx, "Atan", 7, 1, 1);
const static OperatorIdentifier Atanh_9(Domain::ai_onnx, "Atanh", 9, 1, 1);
const static OperatorIdentifier
    AveragePool_1(Domain::ai_onnx, "AveragePool", 1, 1, 1);
const static OperatorIdentifier
    AveragePool_7(Domain::ai_onnx, "AveragePool", 7, 1, 1);
const static OperatorIdentifier
    BatchNormalization_1(Domain::ai_onnx, "BatchNormalization", 1, 5, 5);
const static OperatorIdentifier
    BatchNormalization_6(Domain::ai_onnx, "BatchNormalization", 6, 5, 5);
const static OperatorIdentifier
    BatchNormalization_7(Domain::ai_onnx, "BatchNormalization", 7, 5, 5);
const static OperatorIdentifier
    BatchNormalization_9(Domain::ai_onnx, "BatchNormalization", 9, 5, 5);
const static OperatorIdentifier Cast_1(Domain::ai_onnx, "Cast", 1, 1, 1);
const static OperatorIdentifier Cast_6(Domain::ai_onnx, "Cast", 6, 1, 1);
const static OperatorIdentifier Cast_9(Domain::ai_onnx, "Cast", 9, 1, 1);
const static OperatorIdentifier Ceil_1(Domain::ai_onnx, "Ceil", 1, 1, 1);
const static OperatorIdentifier Ceil_6(Domain::ai_onnx, "Ceil", 6, 1, 1);
const static OperatorIdentifier Clip_1(Domain::ai_onnx, "Clip", 1, 1, 1);
const static OperatorIdentifier Clip_6(Domain::ai_onnx, "Clip", 6, 1, 1);
const static OperatorIdentifier
    Compress_9(Domain::ai_onnx, "Compress", 9, 2, 1);
const static OperatorIdentifier
    Concat_1(Domain::ai_onnx, "Concat", 1, {1, -1}, 1);
const static OperatorIdentifier
    Concat_4(Domain::ai_onnx, "Concat", 4, {1, -1}, 1);
const static OperatorIdentifier
    Constant_1(Domain::ai_onnx, "Constant", 1, 0, 1);
const static OperatorIdentifier
    Constant_9(Domain::ai_onnx, "Constant", 9, 0, 1);
const static OperatorIdentifier
    ConstantOfShape_9(Domain::ai_onnx, "ConstantOfShape", 9, 1, 1);
const static OperatorIdentifier Conv_1(Domain::ai_onnx, "Conv", 1, {2, 3}, 1);
const static OperatorIdentifier
    ConvTranspose_1(Domain::ai_onnx, "ConvTranspose", 1, {2, 3}, 1);
const static OperatorIdentifier Cos_7(Domain::ai_onnx, "Cos", 7, 1, 1);
const static OperatorIdentifier Cosh_9(Domain::ai_onnx, "Cosh", 9, 1, 1);
const static OperatorIdentifier
    DepthToSpace_1(Domain::ai_onnx, "DepthToSpace", 1, 1, 1);
const static OperatorIdentifier Div_1(Domain::ai_onnx, "Div", 1, 2, 1);
const static OperatorIdentifier Div_6(Domain::ai_onnx, "Div", 6, 2, 1);
const static OperatorIdentifier Div_7(Domain::ai_onnx, "Div", 7, 2, 1);
const static OperatorIdentifier Dropout_1(Domain::ai_onnx, "Dropout", 1, 1, 2);
const static OperatorIdentifier Dropout_6(Domain::ai_onnx, "Dropout", 6, 1, 2);
const static OperatorIdentifier Dropout_7(Domain::ai_onnx, "Dropout", 7, 1, 2);
const static OperatorIdentifier Elu_1(Domain::ai_onnx, "Elu", 1, 1, 1);
const static OperatorIdentifier Elu_6(Domain::ai_onnx, "Elu", 6, 1, 1);
const static OperatorIdentifier Equal_1(Domain::ai_onnx, "Equal", 1, 2, 1);
const static OperatorIdentifier Equal_7(Domain::ai_onnx, "Equal", 7, 2, 1);
const static OperatorIdentifier Erf_9(Domain::ai_onnx, "Erf", 9, 1, 1);
const static OperatorIdentifier Exp_1(Domain::ai_onnx, "Exp", 1, 1, 1);
const static OperatorIdentifier Exp_6(Domain::ai_onnx, "Exp", 6, 1, 1);
const static OperatorIdentifier Expand_8(Domain::ai_onnx, "Expand", 8, 2, 1);
const static OperatorIdentifier EyeLike_9(Domain::ai_onnx, "EyeLike", 9, 1, 1);
const static OperatorIdentifier Flatten_1(Domain::ai_onnx, "Flatten", 1, 1, 1);
const static OperatorIdentifier Flatten_9(Domain::ai_onnx, "Flatten", 9, 1, 1);
const static OperatorIdentifier Floor_1(Domain::ai_onnx, "Floor", 1, 1, 1);
const static OperatorIdentifier Floor_6(Domain::ai_onnx, "Floor", 6, 1, 1);
const static OperatorIdentifier GRU_1(Domain::ai_onnx, "GRU", 7, {3, 6}, 2);
const static OperatorIdentifier GRU_3(Domain::ai_onnx, "GRU", 7, {3, 6}, 2);
const static OperatorIdentifier GRU_7(Domain::ai_onnx, "GRU", 7, {3, 6}, 2);
const static OperatorIdentifier Gather_1(Domain::ai_onnx, "Gather", 1, 2, 1);
const static OperatorIdentifier Gemm_1(Domain::ai_onnx, "Gemm", 6, 3, 1);
const static OperatorIdentifier Gemm_6(Domain::ai_onnx, "Gemm", 6, 3, 1);
const static OperatorIdentifier Gemm_7(Domain::ai_onnx, "Gemm", 7, 3, 1);
const static OperatorIdentifier Gemm_9(Domain::ai_onnx, "Gemm", 9, 3, 1);
const static OperatorIdentifier
    GlobalAveragePool_1(Domain::ai_onnx, "GlobalAveragePool", 1, 1, 1);
const static OperatorIdentifier
    GlobalLpPool_1(Domain::ai_onnx, "GlobalLpPool", 1, 1, 1);
const static OperatorIdentifier
    GlobalLpPool_2(Domain::ai_onnx, "GlobalLpPool", 2, 1, 1);
const static OperatorIdentifier
    GlobalMaxPool_1(Domain::ai_onnx, "GlobalMaxPool", 1, 1, 1);
const static OperatorIdentifier Greater_1(Domain::ai_onnx, "Greater", 1, 2, 1);
const static OperatorIdentifier Greater_7(Domain::ai_onnx, "Greater", 7, 2, 1);
const static OperatorIdentifier Greater_9(Domain::ai_onnx, "Greater", 9, 2, 1);
const static OperatorIdentifier
    HardSigmoid_1(Domain::ai_onnx, "HardSigmoid", 1, 1, 1);
const static OperatorIdentifier
    HardSigmoid_6(Domain::ai_onnx, "HardSigmoid", 6, 1, 1);
const static OperatorIdentifier Hardmax_1(Domain::ai_onnx, "Hardmax", 1, 1, 1);
const static OperatorIdentifier
    Identity_1(Domain::ai_onnx, "Identity", 1, 1, 1);
const static OperatorIdentifier If_1(Domain::ai_onnx, "If", 1, 1, -1);
const static OperatorIdentifier
    InstanceNormalization_1(Domain::ai_onnx, "InstanceNormalization", 1, 3, 1);
const static OperatorIdentifier
    InstanceNormalization_6(Domain::ai_onnx, "InstanceNormalization", 6, 3, 1);
const static OperatorIdentifier IsNaN_9(Domain::ai_onnx, "IsNan", 9, 1, 1);
const static OperatorIdentifier LRN_1(Domain::ai_onnx, "LRN", 1, 1, 1);
const static OperatorIdentifier LSTM_1(Domain::ai_onnx, "LSTM", 1, {3, 8}, 3);
const static OperatorIdentifier LSTM_7(Domain::ai_onnx, "LSTM", 7, {3, 8}, 3);
const static OperatorIdentifier
    LeakyRelu_1(Domain::ai_onnx, "LeakyRely", 6, 1, 1);
const static OperatorIdentifier
    LeakyRelu_6(Domain::ai_onnx, "LeakyRely", 6, 1, 1);
const static OperatorIdentifier Less_1(Domain::ai_onnx, "Less", 1, 2, 1);
const static OperatorIdentifier Less_7(Domain::ai_onnx, "Less", 7, 2, 1);
const static OperatorIdentifier Less_9(Domain::ai_onnx, "Less", 9, 2, 1);
const static OperatorIdentifier Log_1(Domain::ai_onnx, "Log", 1, 1, 1);
const static OperatorIdentifier Log_6(Domain::ai_onnx, "Log", 6, 1, 1);
const static OperatorIdentifier
    LogSoftmax_1(Domain::ai_onnx, "LogSoftmax", 1, 1, 1);
const static OperatorIdentifier Loop_1(Domain::ai_onnx, "Loop", 1, {3, -1}, -1);
const static OperatorIdentifier
    LpNormalization_1(Domain::ai_onnx, "LpNormalization", 1, 1, 1);
const static OperatorIdentifier LpPool_1(Domain::ai_onnx, "LpPool", 1, 1, 1);
const static OperatorIdentifier LpPool_2(Domain::ai_onnx, "LpPool", 2, 1, 1);
const static OperatorIdentifier MatMul_1(Domain::ai_onnx, "MatMul", 1, 2, 1);
const static OperatorIdentifier MatMul_9(Domain::ai_onnx, "MatMul", 9, 2, 1);
const static OperatorIdentifier Max_1(Domain::ai_onnx, "Max", 1, {1, -1}, 1);
const static OperatorIdentifier Max_6(Domain::ai_onnx, "Max", 6, {1, -1}, 1);
const static OperatorIdentifier Max_8(Domain::ai_onnx, "Max", 8, {1, -1}, 1);
const static OperatorIdentifier MaxPool_1(Domain::ai_onnx, "MaxPool", 1, 1, 2);
const static OperatorIdentifier MaxPool_8(Domain::ai_onnx, "MaxPool", 8, 1, 2);
const static OperatorIdentifier
    MaxRoiPool_1(Domain::ai_onnx, "MaxRoiPool", 1, 2, 1);
const static OperatorIdentifier
    MaxUnpool_9(Domain::ai_onnx, "MaxUnpool", 9, {2, 3}, 1);
const static OperatorIdentifier Mean_1(Domain::ai_onnx, "Mean", 1, {1, -1}, 1);
const static OperatorIdentifier Mean_6(Domain::ai_onnx, "Mean", 6, {1, -1}, 1);
const static OperatorIdentifier Mean_8(Domain::ai_onnx, "Mean", 8, {1, -1}, 1);
const static OperatorIdentifier Min_1(Domain::ai_onnx, "Min", 1, {1, -1}, 1);
const static OperatorIdentifier Min_6(Domain::ai_onnx, "Min", 6, {1, -1}, 1);
const static OperatorIdentifier Min_8(Domain::ai_onnx, "Min", 8, {1, -1}, 1);
const static OperatorIdentifier Mul_1(Domain::ai_onnx, "Mul", 1, 2, 1);
const static OperatorIdentifier Mul_6(Domain::ai_onnx, "Mul", 6, 2, 1);
const static OperatorIdentifier Mul_7(Domain::ai_onnx, "Mul", 7, 2, 1);
const static OperatorIdentifier
    Multinomial_7(Domain::ai_onnx, "Multinomial", 7, 1, 1);
const static OperatorIdentifier Neg_1(Domain::ai_onnx, "Neg", 1, 1, 1);
const static OperatorIdentifier Neg_6(Domain::ai_onnx, "Neg", 6, 1, 1);
const static OperatorIdentifier Not_1(Domain::ai_onnx, "Not", 1, 1, 1);
const static OperatorIdentifier NonZero_9(Domain::ai_onnx, "NonZero", 9, 1, 1);
const static OperatorIdentifier OneHot_9(Domain::ai_onnx, "OneHot", 9, 3, 1);
const static OperatorIdentifier Or_1(Domain::ai_onnx, "Or", 1, 2, 1);
const static OperatorIdentifier Or_7(Domain::ai_onnx, "Or", 7, 2, 1);
const static OperatorIdentifier PRelu_1(Domain::ai_onnx, "PRelu", 1, 2, 1);
const static OperatorIdentifier PRelu_6(Domain::ai_onnx, "PRelu", 6, 2, 1);
const static OperatorIdentifier PRelu_7(Domain::ai_onnx, "PRelu", 7, 2, 1);
const static OperatorIdentifier PRelu_9(Domain::ai_onnx, "PRelu", 9, 2, 1);
const static OperatorIdentifier Pad_1(Domain::ai_onnx, "Pad", 1, 1, 1);
const static OperatorIdentifier Pad_2(Domain::ai_onnx, "Pad", 2, 1, 1);
const static OperatorIdentifier Pow_1(Domain::ai_onnx, "Pow", 1, 2, 1);
const static OperatorIdentifier Pow_7(Domain::ai_onnx, "Pow", 7, 2, 1);
const static OperatorIdentifier RNN_1(Domain::ai_onnx, "RNN", 1, {3, 6}, 2);
const static OperatorIdentifier RNN_7(Domain::ai_onnx, "RNN", 7, {3, 6}, 2);
const static OperatorIdentifier
    RandomNormal_1(Domain::ai_onnx, "RandomNormal", 1, 0, 1);
const static OperatorIdentifier
    RandomNormalLike_1(Domain::ai_onnx, "RandomNormalLike", 1, 1, 1);
const static OperatorIdentifier
    RandomUniform_1(Domain::ai_onnx, "RandomUniform", 1, 0, 1);
const static OperatorIdentifier
    RandomUniformLike_1(Domain::ai_onnx, "RandomUniformLike", 1, 1, 1);
const static OperatorIdentifier
    Reciprocal_1(Domain::ai_onnx, "Reciprocal", 1, 1, 1);
const static OperatorIdentifier
    Reciprocal_6(Domain::ai_onnx, "Reciprocal", 6, 1, 1);
const static OperatorIdentifier
    ReduceL1_1(Domain::ai_onnx, "ReduceL1", 1, 1, 1);
const static OperatorIdentifier
    ReduceL2_1(Domain::ai_onnx, "ReduceL2", 1, 1, 1);
const static OperatorIdentifier
    ReduceLogSum_1(Domain::ai_onnx, "ReduceLogSum", 1, 1, 1);
const static OperatorIdentifier
    ReduceLogSumExp_1(Domain::ai_onnx, "ReduceLogSumExp", 1, 1, 1);
const static OperatorIdentifier
    ReduceMax_1(Domain::ai_onnx, "ReduceMax", 1, 1, 1);
const static OperatorIdentifier
    ReduceMean_1(Domain::ai_onnx, "ReduceMean", 1, 1, 1);
const static OperatorIdentifier
    ReduceMin_1(Domain::ai_onnx, "ReduceMin", 1, 1, 1);
const static OperatorIdentifier
    ReduceProd_1(Domain::ai_onnx, "ReduceProd", 1, 1, 1);
const static OperatorIdentifier
    ReduceSum_1(Domain::ai_onnx, "ReduceSum", 1, 1, 1);
const static OperatorIdentifier
    ReduceSumSquare_1(Domain::ai_onnx, "ReduceSumSquare", 1, 1, 1);
const static OperatorIdentifier Relu_1(Domain::ai_onnx, "Relu", 1, 1, 1);
const static OperatorIdentifier Relu_6(Domain::ai_onnx, "Relu", 6, 1, 1);
const static OperatorIdentifier Reshape_1(Domain::ai_onnx, "Reshape", 1, 2, 1);
const static OperatorIdentifier Reshape_5(Domain::ai_onnx, "Reshape", 5, 2, 1);
const static OperatorIdentifier Scan_8(Domain::ai_onnx, "Scan", 8, {1, -1}, -1);
const static OperatorIdentifier Scan_9(Domain::ai_onnx, "Scan", 9, {1, -1}, -1);
const static OperatorIdentifier Scatter_9(Domain::ai_onnx, "Scatter", 9, 3, 1);
const static OperatorIdentifier Selu_1(Domain::ai_onnx, "Selu", 1, 1, 1);
const static OperatorIdentifier Selu_6(Domain::ai_onnx, "Selu", 6, 1, 1);
const static OperatorIdentifier Shape_1(Domain::ai_onnx, "Shape", 1, 1, 1);
const static OperatorIdentifier Shrink_9(Domain::ai_onnx, "Shrink", 9, 1, 1);
const static OperatorIdentifier Sigmoid_1(Domain::ai_onnx, "Sigmoid", 1, 1, 1);
const static OperatorIdentifier Sigmoid_6(Domain::ai_onnx, "Sigmoid", 6, 1, 1);
const static OperatorIdentifier Sign_9(Domain::ai_onnx, "Sign", 9, 1, 1);
const static OperatorIdentifier Sin_7(Domain::ai_onnx, "Sin", 7, 1, 1);
const static OperatorIdentifier Sinh_9(Domain::ai_onnx, "Sinh", 9, 1, 1);
const static OperatorIdentifier Size_1(Domain::ai_onnx, "Size", 1, 1, 1);
const static OperatorIdentifier Slice_1(Domain::ai_onnx, "Slice", 1, 1, 1);
const static OperatorIdentifier Softmax_1(Domain::ai_onnx, "Softmax", 1, 1, 1);
const static OperatorIdentifier
    Softplus_1(Domain::ai_onnx, "Softplus", 1, 1, 1);
const static OperatorIdentifier
    Softsign_1(Domain::ai_onnx, "Softsign", 1, 1, 1);
const static OperatorIdentifier
    SpaceToDepth_1(Domain::ai_onnx, "SpaceToDepth", 1, 1, 1);
const static OperatorIdentifier Split_1(Domain::ai_onnx, "Split", 1, 1, -1);
const static OperatorIdentifier Split_2(Domain::ai_onnx, "Split", 2, 1, -1);
const static OperatorIdentifier Sqrt_1(Domain::ai_onnx, "Sqrt", 1, 1, 1);
const static OperatorIdentifier Sqrt_6(Domain::ai_onnx, "Sqrt", 6, 1, 1);
const static OperatorIdentifier Squeeze_1(Domain::ai_onnx, "Squeeze", 1, 1, 1);
const static OperatorIdentifier Sub_1(Domain::ai_onnx, "Sub", 1, 2, 1);
const static OperatorIdentifier Sub_6(Domain::ai_onnx, "Sub", 6, 2, 1);
const static OperatorIdentifier Sub_7(Domain::ai_onnx, "Sub", 7, 2, 1);
const static OperatorIdentifier Sum_1(Domain::ai_onnx, "Sum", 1, {1, -1}, 1);
const static OperatorIdentifier Sum_6(Domain::ai_onnx, "Sum", 6, {1, -1}, 1);
const static OperatorIdentifier Sum_8(Domain::ai_onnx, "Sum", 8, {1, -1}, 1);
const static OperatorIdentifier Tan_7(Domain::ai_onnx, "Tan", 7, 1, 1);
const static OperatorIdentifier Tanh_1(Domain::ai_onnx, "Tanh", 1, 1, 1);
const static OperatorIdentifier Tanh_6(Domain::ai_onnx, "Tanh", 6, 1, 1);
const static OperatorIdentifier Tile_1(Domain::ai_onnx, "Tile", 1, 2, 1);
const static OperatorIdentifier Tile_6(Domain::ai_onnx, "Tile", 6, 2, 1);
const static OperatorIdentifier TopK_1(Domain::ai_onnx, "TopK", 1, 1, 2);
const static OperatorIdentifier
    TfIdfVectorizer_9(Domain::ai_onnx, "TfIdfVectorizer", 9, 1, 1);
const static OperatorIdentifier
    Transpose_1(Domain::ai_onnx, "Transpose", 1, 1, 1);

const static OperatorIdentifier
    Unsqueeze_1(Domain::ai_onnx, "Unsqueeze", 1, 1, 1);
const static OperatorIdentifier
    Upsample_1(Domain::ai_onnx, "Upsample", 7, 2, 1);
const static OperatorIdentifier
    Upsample_7(Domain::ai_onnx, "Upsample", 7, 2, 1);
const static OperatorIdentifier
    Upsample_9(Domain::ai_onnx, "Upsample", 9, 2, 1);
const static OperatorIdentifier Where_9(Domain::ai_onnx, "Where", 9, 3, 1);
const static OperatorIdentifier Xor_1(Domain::ai_onnx, "Xor", 1, 2, 1);
const static OperatorIdentifier Xor_7(Domain::ai_onnx, "Xor", 7, 2, 1);
// experimental
const static OperatorIdentifier ATen_1(Domain::ai_onnx, "ATen", 0, {1, -1}, -1);
const static OperatorIdentifier Affine_1(Domain::ai_onnx, "Affine", 0, 1, 1);
const static OperatorIdentifier
    ConstantFill_1(Domain::ai_onnx, "ConstantFill", 0, {0, 1}, 1);
const static OperatorIdentifier Crop_1(Domain::ai_onnx, "Crop", 0, 1, 1);
const static OperatorIdentifier
    DynamicSlice_1(Domain::ai_onnx, "DynamicSlice", 0, {3, 4}, 1);
const static OperatorIdentifier GRUUnit_1(Domain::ai_onnx, "GRUUnit", 0, 4, 1);
const static OperatorIdentifier
    GivenTensorFill_1(Domain::ai_onnx, "GivenTensorFill", 0, {0, 1}, 1);
const static OperatorIdentifier
    ImageScaler_1(Domain::ai_onnx, "ImageScaler", 0, 1, 1);
const static OperatorIdentifier
    ParametricSoftplus_1(Domain::ai_onnx, "ParametricSoftplus", 0, 1, 1);
const static OperatorIdentifier Scale_1(Domain::ai_onnx, "Scale", 0, 1, 1);
const static OperatorIdentifier
    ScaledTanh_1(Domain::ai_onnx, "ScaledTanh", 0, 1, 1);
const static OperatorIdentifier
    ThresholdedRelu_1(Domain::ai_onnx, "ThresholdedRelu", 0, 1, 1);

const static OperatorIdentifier ArrayFeatureExtractor_1(Domain::ai_onnx_ml,
                                                        "ArrayFeatureExtractor",
                                                        1,
                                                        2,
                                                        1);

} // namespace Operators

// The set of operators in OpSet9. This is to make the code cleaner elsewhere
namespace AiOnnx {
namespace OpSet8 {
// Place holder
}
namespace OpSet9 {
const static OperatorIdentifier Abs         = Operators::Abs_6;
const static OperatorIdentifier Acos        = Operators::Acos_7;
const static OperatorIdentifier Acosh       = Operators::Acosh_9;
const static OperatorIdentifier Add         = Operators::Add_7;
const static OperatorIdentifier And         = Operators::And_7;
const static OperatorIdentifier ArgMax      = Operators::ArgMax_1;
const static OperatorIdentifier ArgMin      = Operators::ArgMin_1;
const static OperatorIdentifier Asin        = Operators::Asin_7;
const static OperatorIdentifier Asinh       = Operators::Asinh_9;
const static OperatorIdentifier Atan        = Operators::Atan_7;
const static OperatorIdentifier Atanh       = Operators::Atanh_9;
const static OperatorIdentifier AveragePool = Operators::AveragePool_7;
const static OperatorIdentifier BatchNormalization =
    Operators::BatchNormalization_9;
const static OperatorIdentifier Cast            = Operators::Cast_6;
const static OperatorIdentifier Ceil            = Operators::Ceil_6;
const static OperatorIdentifier Clip            = Operators::Clip_6;
const static OperatorIdentifier Compress        = Operators::Compress_9;
const static OperatorIdentifier Concat          = Operators::Concat_4;
const static OperatorIdentifier Constant        = Operators::Constant_9;
const static OperatorIdentifier ConstantOfShape = Operators::ConstantOfShape_9;
const static OperatorIdentifier Conv            = Operators::Conv_1;
const static OperatorIdentifier ConvTranspose   = Operators::ConvTranspose_1;
const static OperatorIdentifier Cos             = Operators::Cos_7;
const static OperatorIdentifier Cosh            = Operators::Cosh_9;
const static OperatorIdentifier DepthToSpace    = Operators::DepthToSpace_1;
const static OperatorIdentifier Div             = Operators::Div_7;
const static OperatorIdentifier Dropout         = Operators::Dropout_7;
const static OperatorIdentifier Elu             = Operators::Elu_6;
const static OperatorIdentifier Equal           = Operators::Equal_7;
const static OperatorIdentifier Erf             = Operators::Erf_9;
const static OperatorIdentifier Exp             = Operators::Exp_6;
const static OperatorIdentifier Expand          = Operators::Expand_8;
const static OperatorIdentifier EyeLike         = Operators::EyeLike_9;
const static OperatorIdentifier Flatten         = Operators::Flatten_9;
const static OperatorIdentifier Floor           = Operators::Floor_6;
const static OperatorIdentifier GRU             = Operators::GRU_7;
const static OperatorIdentifier Gather          = Operators::Gather_1;
const static OperatorIdentifier Gemm            = Operators::Gemm_9;
const static OperatorIdentifier GlobalAveragePool =
    Operators::GlobalAveragePool_1;
const static OperatorIdentifier GlobalLpPool  = Operators::GlobalLpPool_2;
const static OperatorIdentifier GlobalMaxPool = Operators::GlobalMaxPool_1;
const static OperatorIdentifier Greater       = Operators::Greater_9;
const static OperatorIdentifier HardSigmoid   = Operators::HardSigmoid_6;
const static OperatorIdentifier Hardmax       = Operators::Hardmax_1;
const static OperatorIdentifier Identity      = Operators::Identity_1;
const static OperatorIdentifier If            = Operators::If_1;
const static OperatorIdentifier InstanceNormalization =
    Operators::InstanceNormalization_6;
const static OperatorIdentifier IsNaN           = Operators::IsNaN_9;
const static OperatorIdentifier LRN             = Operators::LRN_1;
const static OperatorIdentifier LSTM            = Operators::LSTM_7;
const static OperatorIdentifier LeakyRelu       = Operators::LeakyRelu_6;
const static OperatorIdentifier Less            = Operators::Less_9;
const static OperatorIdentifier Log             = Operators::Log_6;
const static OperatorIdentifier LogSoftmax      = Operators::LogSoftmax_1;
const static OperatorIdentifier Loop            = Operators::Loop_1;
const static OperatorIdentifier LpNormalization = Operators::LpNormalization_1;
const static OperatorIdentifier LpPool          = Operators::LpPool_2;
const static OperatorIdentifier MatMul          = Operators::MatMul_9;
const static OperatorIdentifier Max             = Operators::Max_8;
const static OperatorIdentifier MaxPool         = Operators::MaxPool_8;
const static OperatorIdentifier MaxRoiPool      = Operators::MaxRoiPool_1;
const static OperatorIdentifier MaxUnpool       = Operators::MaxUnpool_9;
const static OperatorIdentifier Mean            = Operators::Mean_8;
const static OperatorIdentifier Min             = Operators::Min_8;
const static OperatorIdentifier Mul             = Operators::Mul_7;
const static OperatorIdentifier Multinomial     = Operators::Multinomial_7;
const static OperatorIdentifier Neg             = Operators::Neg_6;
const static OperatorIdentifier Not             = Operators::Not_1;
const static OperatorIdentifier OneHot          = Operators::OneHot_9;
const static OperatorIdentifier Or              = Operators::Or_7;
const static OperatorIdentifier PRelu           = Operators::PRelu_9;
const static OperatorIdentifier Pad             = Operators::Pad_2;
const static OperatorIdentifier Pow             = Operators::Pow_7;
const static OperatorIdentifier RNN             = Operators::RNN_7;
const static OperatorIdentifier RandomNormal    = Operators::RandomNormal_1;
const static OperatorIdentifier RandomNormalLike =
    Operators::RandomNormalLike_1;
const static OperatorIdentifier RandomUniform = Operators::RandomUniform_1;
const static OperatorIdentifier RandomUniformLike =
    Operators::RandomUniformLike_1;
const static OperatorIdentifier Reciprocal      = Operators::Reciprocal_6;
const static OperatorIdentifier ReduceL1        = Operators::ReduceL1_1;
const static OperatorIdentifier ReduceL2        = Operators::ReduceL2_1;
const static OperatorIdentifier ReduceLogSum    = Operators::ReduceLogSum_1;
const static OperatorIdentifier ReduceLogSumExp = Operators::ReduceLogSumExp_1;
const static OperatorIdentifier ReduceMax       = Operators::ReduceMax_1;
const static OperatorIdentifier ReduceMean      = Operators::ReduceMean_1;
const static OperatorIdentifier ReduceMin       = Operators::ReduceMin_1;
const static OperatorIdentifier ReduceProd      = Operators::ReduceProd_1;
const static OperatorIdentifier ReduceSum       = Operators::ReduceSum_1;
const static OperatorIdentifier ReduceSumSquare = Operators::ReduceSumSquare_1;
const static OperatorIdentifier Relu            = Operators::Relu_6;
const static OperatorIdentifier Reshape         = Operators::Reshape_5;
const static OperatorIdentifier Scan            = Operators::Scan_9;
const static OperatorIdentifier Scatter         = Operators::Scatter_9;
const static OperatorIdentifier Selu            = Operators::Selu_6;
const static OperatorIdentifier Shape           = Operators::Shape_1;
const static OperatorIdentifier Shrink          = Operators::Shrink_9;
const static OperatorIdentifier Sigmoid         = Operators::Sigmoid_6;
const static OperatorIdentifier Sign            = Operators::Sign_9;
const static OperatorIdentifier Sin             = Operators::Sin_7;
const static OperatorIdentifier Sinh            = Operators::Sinh_9;
const static OperatorIdentifier Size            = Operators::Size_1;
const static OperatorIdentifier Slice           = Operators::Slice_1;
const static OperatorIdentifier Softmax         = Operators::Softmax_1;
const static OperatorIdentifier Softplus        = Operators::Softplus_1;
const static OperatorIdentifier Softsign        = Operators::Softsign_1;
const static OperatorIdentifier SpaceToDepth    = Operators::SpaceToDepth_1;
const static OperatorIdentifier Split           = Operators::Split_2;
const static OperatorIdentifier Sqrt            = Operators::Sqrt_6;
const static OperatorIdentifier Squeeze         = Operators::Squeeze_1;
const static OperatorIdentifier Sub             = Operators::Sub_7;
const static OperatorIdentifier Sum             = Operators::Sum_8;
const static OperatorIdentifier Tan             = Operators::Tan_7;
const static OperatorIdentifier Tanh            = Operators::Tanh_6;
const static OperatorIdentifier Tile            = Operators::Tile_6;
const static OperatorIdentifier TopK            = Operators::TopK_1;
const static OperatorIdentifier Transpose       = Operators::Transpose_1;
const static OperatorIdentifier Unsqueeze       = Operators::Unsqueeze_1;
const static OperatorIdentifier Upsample        = Operators::Upsample_9;
const static OperatorIdentifier Xor             = Operators::Xor_7;

// experimental - no specific version
const static OperatorIdentifier ATen            = Operators::ATen_1;
const static OperatorIdentifier Affine          = Operators::Affine_1;
const static OperatorIdentifier ConstantFill    = Operators::ConstantFill_1;
const static OperatorIdentifier Crop            = Operators::Crop_1;
const static OperatorIdentifier DynamicSlice    = Operators::DynamicSlice_1;
const static OperatorIdentifier GRUUnit         = Operators::GRUUnit_1;
const static OperatorIdentifier GivenTensorFill = Operators::GivenTensorFill_1;
const static OperatorIdentifier ImageScaler     = Operators::ImageScaler_1;
const static OperatorIdentifier ParametricSoftplus =
    Operators::ParametricSoftplus_1;
const static OperatorIdentifier Scale           = Operators::Scale_1;
const static OperatorIdentifier ScaledTanh      = Operators::ScaledTanh_1;
const static OperatorIdentifier ThresholdedRelu = Operators::ThresholdedRelu_1;

} // namespace OpSet9
} // namespace AiOnnx

namespace AiOnnxMl {
namespace OpSet1 {
const static OperatorIdentifier ArrayFeatureExtractor =
    Operators::ArrayFeatureExtractor_1;
}
} // namespace AiOnnxMl

namespace GradOperators {
const static AiGraphcoreOpIdV1 AbsGrad("AbsGrad");
// constexpr static char Acos[] = "Acos";
// constexpr static char Acosh[] = "Acosh";
const static AiGraphcoreOpIdV1 AddArg0Grad("AddArg0Grad");
const static AiGraphcoreOpIdV1 AddArg1Grad("AddArg1Grad");
// constexpr static char And[] = "And";
// constexpr static char ArgMax[] = "ArgMax";
// constexpr static char ArgMin[] = "ArgMin";
// constexpr static char Asin[] = "Asin";
// constexpr static char Asinh[] = "Asinh";
// constexpr static char Atan[] = "Atan";
// constexpr static char Atanh[] = "Atanh";
const static AiGraphcoreOpIdV1 AveragePoolGrad("AveragePoolGrad");
const static AiGraphcoreOpIdV1 BatchNormalizationGrad("BatchNormalizationGrad");
const static AiGraphcoreOpIdV1 CastGrad("CastGrad");
// constexpr static char Ceil[] = "Ceil";
// constexpr static char Clip[] = "Clip";
// constexpr static char Compress[] = "Compress";
const static AiGraphcoreOpIdV1 ConcatGrad("ConcatGrad");
// constexpr static char Constant[] = "Constant";
// constexpr static char ConstantLike[] = "ConstantLike";
const static AiGraphcoreOpIdV1 ConvDataGrad("ConvDataGrad");
const static AiGraphcoreOpIdV1 ConvWeightsGrad("ConvWeightsGrad");
// constexpr static char ConvTranspose[] = "ConvTranspose";
const static AiGraphcoreOpIdV1 CosGrad("CosGrad");
// constexpr static char Cosh[] = "Cosh"
// constexpr static char DepthToSpace[] = "DepthToSpace";
const static AiGraphcoreOpIdV1 DivArg0Grad("DivArg0Grad");
const static AiGraphcoreOpIdV1 DivArg1Grad("DivArg1Grad");
// constexpr static char Dropout[] = "DropOut";
// constexpr static char Elu[] = "Elu";
// constexpr static char Equal[] = "Equal";
// constexpr static char Erf[] = "Erf";
const static AiGraphcoreOpIdV1 ExpGrad("ExpGrad");
// constexpr static char Expand[] = "Expand";
// constexpr static char EyeLike[] = "EyeLike";
const static AiGraphcoreOpIdV1 FlattenGrad("FlattenGrad");
// constexpr static char Floor[] = "Floor";
// constexpr static char GRU[] = "GRU";
const static AiGraphcoreOpIdV1 GatherGrad("GatherGrad");
// constexpr static char Gemm[] = "Gemm";
const static AiGraphcoreOpIdV1 GlobalAveragePoolGrad("GlobalAveragePoolGrad");
// constexpr static char GlobalLpPool[] = "GlobalLpPool";
const static AiGraphcoreOpIdV1 GlobalMaxPoolGrad("GlobalMaxPoolGrad");
// constexpr static char Greater[] = "Greater";
const static AiGraphcoreOpIdV1 GroupNormalizationGrad("GroupNormalizationGrad");
// constexpr static char HardSigmoid[] = "HardSigmoid";
// constexpr static char Hardmax[] = "Hardmax";
const static AiGraphcoreOpIdV1 IdentityGrad("IdentityGrad");
// constexpr static char If[] = "If";
const static AiGraphcoreOpIdV1
    InstanceNormalizationGrad("InstanceNormalizationGrad");
// constexpr static char IsNaN[] = "IsNan";
// constexpr static char LRN[] = "LRN";
const static AiGraphcoreOpIdV1 LSTMGrad("LSTMGrad");
// constexpr static char LeakyRelu[] = "LeakyRely";
// constexpr static char Less[] = "Less";
const static AiGraphcoreOpIdV1 LogGrad("LogGrad");
// constexpr static char LogSoftmax[] = "LogSoftmax";
// constexpr static char Loop[] = "Loop";
// constexpr static char LpNormalization[] = "LpNormalization";
// constexpr static char LpPool[] = "LpPool";
const static AiGraphcoreOpIdV1 MatMulLhsGrad("MatMulLhsGrad");
const static AiGraphcoreOpIdV1 MatMulRhsGrad("MatMulRhsGrad");
const static AiGraphcoreOpIdV1 MaxArgGrad("MaxArgGrad");
const static AiGraphcoreOpIdV1 MaxPoolGrad("MaxPoolGrad");
// constexpr static char MaxRoiPool[] = "MaxRoiPooL";
// constexpr static char MaxUnpool[] = "MaxUnpool";
const static AiGraphcoreOpIdV1 MeanArgGrad("MeanArgGrad");
const static AiGraphcoreOpIdV1 MinArgGrad("MinArgGrad");
const static AiGraphcoreOpIdV1 MulArg0Grad("MulArg0Grad");
const static AiGraphcoreOpIdV1 MulArg1Grad("MulArg1Grad");
// constexpr static char Multinomial[] = "Multinomial";
const static AiGraphcoreOpIdV1 NegGrad("NegGrad");
// constexpr static char Not[] = "Not";
const static AiGraphcoreOpIdV1 OneHotGrad("OneHot");
// constexpr static char Or[] = "Or";
// constexpr static char PRelu[] = "PRelu";
const static AiGraphcoreOpIdV1 PadGrad("PadGrad");
// constexpr static char Pow[] = "Pow";
// constexpr static char RNN[] = "RNN";
// constexpr static char RandomNormal[] = "RandomNormal";
// constexpr static char RandomNormalLike[] = "RandomNormalLike";
// constexpr static char RandomUniform[] = "RandomUniform";
// constexpr static char RandomUniformLike[] = "RandomUniformLike";
const static AiGraphcoreOpIdV1 ReciprocalGrad("ReciprocalGrad");
// constexpr static char ReduceL1[] = "ReduceL1";
// constexpr static char ReduceL2[] = "ReduceL2";
// constexpr static char ReduceLogSum[] = "ReduceLogSum";
// constexpr static char ReduceLogSumExp[] = "ReduceLogSumExp";
// constexpr static char ReduceMax[] = "ReduceMax";
// constexpr static char ReduceMean[] = "ReduceMean";
// constexpr static char ReduceMin[] = "ReduceMin";
// constexpr static char ReduceProd[] = "ReduceProd";
const static AiGraphcoreOpIdV1 ReduceSumGrad("ReduceSumGrad");
// constexpr static char ReduceSumSquare[] = "ReduceSumSquare";
const static AiGraphcoreOpIdV1 ReluGrad("ReluGrad");
const static AiGraphcoreOpIdV1 ReshapeGrad("ReshapeGrad");
// constexpr static char Scan[] = "Scan";
const static AiGraphcoreOpIdV1 ScatterDataGrad("ScatterDataGrad");
const static AiGraphcoreOpIdV1 ScatterUpdateGrad("ScatterUpdateGrad");
// constexpr static char Selu[] = "Selu";
// constexpr static char Shape[] = "Shape";
const static AiGraphcoreOpIdV1 SigmoidGrad("SigmoidGrad");
const static AiGraphcoreOpIdV1 SignGrad("SignGrad");
const static AiGraphcoreOpIdV1 SinGrad("SinGrad");
// constexpr static char Sinh[] = "Sinh";
// constexpr static char Size[] = "Size";
const static AiGraphcoreOpIdV1 SliceGrad("SliceGrad");
const static AiGraphcoreOpIdV1 SoftmaxGrad("SoftmaxGrad");
// constexpr static char Softplus[] = "Softplus";
// constexpr static char Softsign[] = "Softsign";
// constexpr static char SpaceToDepth[] = "SpaceToDepth";
// constexpr static char Split[] = "Split";
const static AiGraphcoreOpIdV1 SqrtGrad("SqrtGrad");
const static AiGraphcoreOpIdV1 SqueezeGrad("SqueezeGrad");
const static AiGraphcoreOpIdV1 SubArg0Grad("SubArg0Grad");
const static AiGraphcoreOpIdV1 SubArg1Grad("SubArg1Grad");
const static AiGraphcoreOpIdV1 SumArgGrad("SumArgGrad");
// constexpr static char Tan[] = "Tan";
const static AiGraphcoreOpIdV1 TanhGrad("TanhGrad");
// constexpr static char Tile[] = "Tile";
// constexpr static char TopK[] = "TopK";
const static AiGraphcoreOpIdV1 TransposeGrad("TransposeGrad");
const static AiGraphcoreOpIdV1 UnsqueezeGrad("UnsqueezeGrad");
// constexpr static char Upsample[] = "Upsample";
// constexpr static char Xor[] = "Xor";
// // experimental
// constexpr static char ATen[] = "ATen";
// constexpr static char Affine[] = "Affine";
// constexpr static char ConstantFill[] = "ConstantFill";
// constexpr static char Crop[] = "Crop";
// constexpr static char DynamicSlice[] = "DynamicSlice";
// constexpr static char GRUUnit[] = "GRUUnit";
// constexpr static char GivenTensorFill[] = "GivenTensorFill";
// constexpr static char ImageScaler[] = "ImageScaler";
// constexpr static char ParametricSoftplus[] = "ParametricSoftplus";
const static AiGraphcoreOpIdV1 ScaleGrad("ScaleGrad");
// constexpr static char ScaledTanh[] = "ScaledTanh";
// constexpr static char ThresholdedRelu[] = "ThresholdedRelu";

} // namespace GradOperators

namespace CustomOperators {
const static AiGraphcoreOpIdV1 AddBias("AddBias");
const static AiGraphcoreOpIdV1 ConcatInplace("ConcatInplace");
const static AiGraphcoreOpIdV1 SliceInplace("SliceInplace");
const static AiGraphcoreOpIdV1 ConcatGradInplace("ConcatInplace");
const static AiGraphcoreOpIdV1 Subsample_1("Subsample", 1, 1);
const static AiGraphcoreOpIdV1 Square("Square");
const static AiGraphcoreOpIdV1 GroupNormalization_1("GroupNormalization", 3, 3);

const static AiGraphcoreOpIdV1 ReluInplace("ReluInplace");
const static AiGraphcoreOpIdV1 ScaleInplace("ScaleInplace");
const static AiGraphcoreOpIdV1 ExpInplace("ExpInplace");
const static AiGraphcoreOpIdV1 L1("L1");
const static AiGraphcoreOpIdV1 Nll("Nll");

const static AiGraphcoreOpIdV1 IpuCopy("IpuCopy");

const static AiGraphcoreOpIdV1 SgdVarUpdate("SGDVarUpdate");
const static AiGraphcoreOpIdV1 ConstSgdVarUpdate("ConstSGDVarUpdate");
const static AiGraphcoreOpIdV1 CopyVarUpdate("CopyVarUpdate");
const static AiGraphcoreOpIdV1 FlattenAlias("FlattenAlias");

const static AiGraphcoreOpIdV1 ConvFlipWeights("ConvFlipWeights");

const static AiGraphcoreOpIdV1 Subgraph("Subgraph");

} // namespace CustomOperators

namespace AiGraphcore {
namespace OpSet1 {
const static OperatorIdentifier Subsample = CustomOperators::Subsample_1;
const static OperatorIdentifier GroupNormalization =
    CustomOperators::GroupNormalization_1;
} // namespace OpSet1
} // namespace AiGraphcore

namespace CustomGradOperators {
const static AiGraphcoreOpIdV1 AddBiasBiasGrad("AddBiasBiasGrad");
const static AiGraphcoreOpIdV1 AddBiasDataGrad("AddBiasDataGrad");
const static AiGraphcoreOpIdV1 ConstantLike("ConstantLike");
const static AiGraphcoreOpIdV1 SoftmaxGradDirect("SoftmaxGradDirect");
const static AiGraphcoreOpIdV1 SubsampleGrad("SubsampleGrad");
const static AiGraphcoreOpIdV1 L1Grad("L1Grad");
const static AiGraphcoreOpIdV1 NllGrad("NllGrad");
} // namespace CustomGradOperators
} // namespace Onnx

} // namespace poponnx

#endif
