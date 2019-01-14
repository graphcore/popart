#ifndef GUARD_NEURALNET_OPIDENTIFIER_HPP
#define GUARD_NEURALNET_OPIDENTIFIER_HPP

#include <poponnx/attributes.hpp>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>

namespace poponnx {

namespace Domain {
const static char *ai_onnx      = "ai.onnx";
const static char *ai_graphcore = "ai.graphcore";
} // namespace Domain

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

  bool operator==(const OperatorIdentifier &rhs) {
    return (domain == rhs.domain && type == rhs.type && version == rhs.version);
  }

  bool operator!=(const OperatorIdentifier &rhs) { return !(*this == rhs); }
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

// FFS There should be a way of generating this list of operators from the md
// files
namespace Onnx {

// When we have to support versioning suggest the following
// const static OperatorIdentifier Add_9(Domain::ai_onnx, "Add", 9);
// const static OperatorIdentifier Add = Add_9;

namespace Operators {
const static AiOnnxOperatorIdentifierV9 Abs("Abs", 1, 1);
const static AiOnnxOperatorIdentifierV9 Acos("Acos", 1, 1);
const static AiOnnxOperatorIdentifierV9 Acosh("Acosh", 1, 1);
const static AiOnnxOperatorIdentifierV9 Add("Add", 2, 1);

const static AiOnnxOperatorIdentifierV9 And("And", 2, 1);
const static AiOnnxOperatorIdentifierV9 ArgMax("ArgMax", 1, 1);
const static AiOnnxOperatorIdentifierV9 ArgMin("ArgMin", 1, 1);
const static AiOnnxOperatorIdentifierV9 Asin("Asin", 1, 1);
const static AiOnnxOperatorIdentifierV9 Asinh("Asinh", 1, 1);
const static AiOnnxOperatorIdentifierV9 Atan("Atan", 1, 1);
const static AiOnnxOperatorIdentifierV9 Atanh("Atanh", 1, 1);
const static AiOnnxOperatorIdentifierV9 AveragePool("AveragePool", 1, 1);
const static AiOnnxOperatorIdentifierV9
    BatchNormalization("BatchNormalization", 5, 5);
const static AiOnnxOperatorIdentifierV9 Cast("Cast", 1, 1);
const static AiOnnxOperatorIdentifierV9 Ceil("Ceil", 1, 1);
const static AiOnnxOperatorIdentifierV9 Clip("Clip", 1, 1);
const static AiOnnxOperatorIdentifierV9 Compress("Compress", 2, 1);
const static AiOnnxOperatorIdentifierV9 Concat("Concat", {1, -1}, 1);
const static AiOnnxOperatorIdentifierV9 Constant("Constant", 0, 1);
const static AiOnnxOperatorIdentifierV9 ConstantLike("ConstantLike");
const static AiOnnxOperatorIdentifierV9 Conv("Conv", {2, 3}, 1);
const static AiOnnxOperatorIdentifierV9
    ConvTranspose("ConvTranspose", {2, 3}, 1);
const static AiOnnxOperatorIdentifierV9 Cos("Cos", 1, 1);
const static AiOnnxOperatorIdentifierV9 Cosh("Cosh", 1, 1);
const static AiOnnxOperatorIdentifierV9 DepthToSpace("DepthToSpace", 1, 1);
const static AiOnnxOperatorIdentifierV9 Div("Div", 2, 1);
const static AiOnnxOperatorIdentifierV9 Dropout("DropOut", 1, 2);
const static AiOnnxOperatorIdentifierV9 Elu("Elu", 1, 1);
const static AiOnnxOperatorIdentifierV9 Equal("Equal", 2, 1);
const static AiOnnxOperatorIdentifierV9 Erf("Erf", 1, 1);
const static AiOnnxOperatorIdentifierV9 Exp("Exp", 1, 1);
const static AiOnnxOperatorIdentifierV9 Expand("Expand", 2, 1);
const static AiOnnxOperatorIdentifierV9 EyeLike("EyeLike", 1, 1);
const static AiOnnxOperatorIdentifierV9 Flatten("Flatten", 1, 1);
const static AiOnnxOperatorIdentifierV9 Floor("Floor", 1, 1);
const static AiOnnxOperatorIdentifierV9 GRU("GRU", {3, 6}, 2);
const static AiOnnxOperatorIdentifierV9 Gather("Gather", 2, 1);
const static AiOnnxOperatorIdentifierV9 Gemm("Gemm", 3, 1);
const static AiOnnxOperatorIdentifierV9
    GlobalAveragePool("GlobalAveragePool", 1, 1);
const static AiOnnxOperatorIdentifierV9 GlobalLpPool("GlobalLpPool", 1, 1);
const static AiOnnxOperatorIdentifierV9 GlobalMaxPool("GlobalMaxPool", 1, 1);
const static AiOnnxOperatorIdentifierV9 Greater("Greater", 2, 1);
const static AiOnnxOperatorIdentifierV9 HardSigmoid("HardSigmoid", 1, 1);
const static AiOnnxOperatorIdentifierV9 Hardmax("Hardmax", 1, 1);
const static AiOnnxOperatorIdentifierV9 Identity("Identity", 1, 1);
const static AiOnnxOperatorIdentifierV9 If("If", 1, -1);
const static AiOnnxOperatorIdentifierV9
    InstanceNormalization("InstanceNormalization", 3, 1);
const static AiOnnxOperatorIdentifierV9 IsNaN("IsNan", 1, 1);
const static AiOnnxOperatorIdentifierV9 LRN("LRN", 1, 1);
const static AiOnnxOperatorIdentifierV9 LSTM("LSTM", {3, 8}, 3);
const static AiOnnxOperatorIdentifierV9 LeakyRelu("LeakyRely", 1, 1);
const static AiOnnxOperatorIdentifierV9 Less("Less", 2, 1);
const static AiOnnxOperatorIdentifierV9 Log("Log", 1, 1);
const static AiOnnxOperatorIdentifierV9 LogSoftmax("LogSoftmax", 1, 1);
const static AiOnnxOperatorIdentifierV9 Loop("Loop", {3, -1}, -1);
const static AiOnnxOperatorIdentifierV9
    LpNormalization("LpNormalization", 1, 1);
const static AiOnnxOperatorIdentifierV9 LpPool("LpPool", 1, 1);
const static AiOnnxOperatorIdentifierV9 MatMul("MatMul", 2, 1);
const static AiOnnxOperatorIdentifierV9 Max("Max", {1, -1}, 1);
const static AiOnnxOperatorIdentifierV9 MaxPool("MaxPool", 1, 2);
const static AiOnnxOperatorIdentifierV9 MaxRoiPool("MaxRoiPool", 2, 1);
const static AiOnnxOperatorIdentifierV9 MaxUnpool("MaxUnpool", {2, 3}, 1);
const static AiOnnxOperatorIdentifierV9 Mean("Mean", {1, -1}, 1);
const static AiOnnxOperatorIdentifierV9 Min("Min", {1, -1}, 1);
const static AiOnnxOperatorIdentifierV9 Mul("Mul", 2, 1);
const static AiOnnxOperatorIdentifierV9 Multinomial("Multinomial", 1, 1);
const static AiOnnxOperatorIdentifierV9 Neg("Neg", 1, 1);
const static AiOnnxOperatorIdentifierV9 Not("Not", 1, 1);
const static AiOnnxOperatorIdentifierV9 OneHot("OneHot", 3, 1);
const static AiOnnxOperatorIdentifierV9 Or("Or", 2, 1);
const static AiOnnxOperatorIdentifierV9 PRelu("PRelu", 2, 1);
const static AiOnnxOperatorIdentifierV9 Pad("Pad", 1, 1);
const static AiOnnxOperatorIdentifierV9 Pow("Pow", 2, 1);
const static AiOnnxOperatorIdentifierV9 RNN("RNN", {3, 6}, 2);
const static AiOnnxOperatorIdentifierV9 RandomNormal("RandomNormal", 0, 1);
const static AiOnnxOperatorIdentifierV9
    RandomNormalLike("RandomNormalLike", 1, 1);
const static AiOnnxOperatorIdentifierV9 RandomUniform("RandomUniform", 0, 1);
const static AiOnnxOperatorIdentifierV9
    RandomUniformLike("RandomUniformLike", 1, 1);
const static AiOnnxOperatorIdentifierV9 Reciprocal("Reciprocal", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceL1("ReduceL1", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceL2("ReduceL2", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceLogSum("ReduceLogSum", 1, 1);
const static AiOnnxOperatorIdentifierV9
    ReduceLogSumExp("ReduceLogSumExp", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceMax("ReduceMax", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceMean("ReduceMean", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceMin("ReduceMin", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceProd("ReduceProd", 1, 1);
const static AiOnnxOperatorIdentifierV9 ReduceSum("ReduceSum", 1, 1);
const static AiOnnxOperatorIdentifierV9
    ReduceSumSquare("ReduceSumSquare", 1, 1);
const static AiOnnxOperatorIdentifierV9 Relu("Relu", 1, 1);
const static AiOnnxOperatorIdentifierV9 Reshape("Reshape", 2, 1);
const static AiOnnxOperatorIdentifierV9 Scan("Scan", {1, -1}, -1);
const static AiOnnxOperatorIdentifierV9 Scatter("Scatter", 3, 1);
const static AiOnnxOperatorIdentifierV9 Selu("Selu", 1, 1);
const static AiOnnxOperatorIdentifierV9 Shape("Shape", 1, 1);
const static AiOnnxOperatorIdentifierV9 Sigmoid("Sigmoid", 1, 1);
const static AiOnnxOperatorIdentifierV9 Sign("Sign", 1, 1);
const static AiOnnxOperatorIdentifierV9 Sin("Sin", 1, 1);
const static AiOnnxOperatorIdentifierV9 Sinh("Sinh", 1, 1);
const static AiOnnxOperatorIdentifierV9 Size("Size", 1, 1);
const static AiOnnxOperatorIdentifierV9 Slice("Slice", 1, 1);
const static AiOnnxOperatorIdentifierV9 Softmax("Softmax", 1, 1);
const static AiOnnxOperatorIdentifierV9 Softplus("Softplus", 1, 1);
const static AiOnnxOperatorIdentifierV9 Softsign("Softsign", 1, 1);
const static AiOnnxOperatorIdentifierV9 SpaceToDepth("SpaceToDepth", 1, 1);
const static AiOnnxOperatorIdentifierV9 Split("Split", 1, -1);
const static AiOnnxOperatorIdentifierV9 Sqrt("Sqrt", 1, 1);
const static AiOnnxOperatorIdentifierV9 Squeeze("Squeeze", 1, 1);
const static AiOnnxOperatorIdentifierV9 Sub("Sub", 2, 1);
const static AiOnnxOperatorIdentifierV9 Sum("Sum", {1, -1}, 1);
const static AiOnnxOperatorIdentifierV9 Tan("Tan", 1, 1);
const static AiOnnxOperatorIdentifierV9 Tanh("Tanh", 1, 1);
const static AiOnnxOperatorIdentifierV9 Tile("Tile", 2, 1);
const static AiOnnxOperatorIdentifierV9 TopK("TopK", 1, 2);
const static AiOnnxOperatorIdentifierV9 Transpose("Transpose", 1, 1);
const static AiOnnxOperatorIdentifierV9 Unsqueeze("Unsqueeze", 1, 1);
const static AiOnnxOperatorIdentifierV9 Upsample("Upsample", 2, 1);
const static AiOnnxOperatorIdentifierV9 Xor("Xor", 2, 1);
// experimental
const static AiOnnxOperatorIdentifierV9 ATen("ATen", {1, -1}, -1);
const static AiOnnxOperatorIdentifierV9 Affine("Affine", 1, 1);
const static AiOnnxOperatorIdentifierV9 ConstantFill("ConstantFill", {0, 1}, 1);
const static AiOnnxOperatorIdentifierV9 Crop("Crop", 1, 1);
const static AiOnnxOperatorIdentifierV9 DynamicSlice("DynamicSlice", {3, 4}, 1);
const static AiOnnxOperatorIdentifierV9 GRUUnit("GRUUnit", 4, 1);
const static AiOnnxOperatorIdentifierV9
    GivenTensorFill("GivenTensorFill", {0, 1}, 1);
const static AiOnnxOperatorIdentifierV9 ImageScaler("ImageScaler", 1, 1);
const static AiOnnxOperatorIdentifierV9
    ParametricSoftplus("ParametricSoftplus", 1, 1);
const static AiOnnxOperatorIdentifierV9 Scale("Scale", 1, 1);
const static AiOnnxOperatorIdentifierV9 ScaledTanh("ScaledTanh", 1, 1);
const static AiOnnxOperatorIdentifierV9
    ThresholdedRelu("ThresholdedRelu", 1, 1);
} // namespace Operators

namespace GradOperators {
// constexpr static char Abs[] = "Abs";
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
// constexpr static char Cast[] = "Cast";
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
// constexpr static char Flatten[] = "Flatten";
// constexpr static char Floor[] = "Floor";
// constexpr static char GRU[] = "GRU";
const static AiGraphcoreOpIdV1 GatherGrad("GatherGrad");
// constexpr static char Gemm[] = "Gemm";
// constexpr static char GlobalAveragePool[] = "GlobalAveragePool";
// constexpr static char GlobalLpPool[] = "GlobalLpPool";
// constexpr static char GlobalMaxPool[] = "GlobalMaxPool";
// constexpr static char Greater[] = "Greater";
// constexpr static char HardSigmoid[] = "HardSigmoid";
// constexpr static char Hardmax[] = "Hardmax";
const static AiGraphcoreOpIdV1 IdentityGrad("IdentityGrad");
// constexpr static char If[] = "If";
// constexpr static char InstanceNormalization[] = "InstanceNormalization";
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
// constexpr static char Max[] = "Max";
const static AiGraphcoreOpIdV1 MaxPoolGrad("MaxPoolGrad");
// constexpr static char MaxRoiPool[] = "MaxRoiPooL";
// constexpr static char MaxUnpool[] = "MaxUnpool";
// constexpr static char Mean[] = "Mean";
// constexpr static char Min[] = "Min";
const static AiGraphcoreOpIdV1 MulArg0Grad("MulArg0Grad");
const static AiGraphcoreOpIdV1 MulArg1Grad("MulArg1Grad");
// constexpr static char Multinomial[] = "Multinomial";
const static AiGraphcoreOpIdV1 NegGrad("NegGrad");
// constexpr static char Not[] = "Not";
// constexpr static char OneHot[] = "OneHot";
// constexpr static char Or[] = "Or";
// constexpr static char PRelu[] = "PRelu";
// constexpr static char Pad[] = "Pad";
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
// constexpr static char Sign[] = "Sign";
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
// constexpr static char Sum[] = "Sum";
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
const static AiGraphcoreOpIdV1 ConcatGradInplace("ConcatInplace");
const static AiGraphcoreOpIdV1 Subsample("Subsample", 1, 1);
const static AiGraphcoreOpIdV1 Square("Square");

const static AiGraphcoreOpIdV1 ReluInplace("ReluInplace");
const static AiGraphcoreOpIdV1 L1("L1");
const static AiGraphcoreOpIdV1 Nll("Nll");

const static AiGraphcoreOpIdV1 IpuCopy("IpuCopy");

const static AiGraphcoreOpIdV1 SgdVarUpdate("SGDVarUpdate");
const static AiGraphcoreOpIdV1 ConstSgdVarUpdate("ConstSGDVarUpdate");
} // namespace CustomOperators

namespace CustomGradOperators {
const static AiGraphcoreOpIdV1 AddBiasBiasGrad("AddBiasBiasGrad");
const static AiGraphcoreOpIdV1 AddBiasDataGrad("AddBiasDataGrad");
const static AiGraphcoreOpIdV1 SoftmaxGradDirect("SoftmaxGradDirect");
const static AiGraphcoreOpIdV1 SubsampleGrad("SubsampleGrad");
const static AiGraphcoreOpIdV1 L1Grad("L1Grad");
const static AiGraphcoreOpIdV1 NllGrad("NllGrad");
} // namespace CustomGradOperators
} // namespace Onnx

} // namespace poponnx

#endif
