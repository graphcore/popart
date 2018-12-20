#ifndef GUARD_NEURALNET_OPIDENTIFIER_HPP
#define GUARD_NEURALNET_OPIDENTIFIER_HPP

#include <poponnx/attributes.hpp>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>

namespace poponnx {

// using OpDomain = std::string;
using OpType    = std::string;
using OpVersion = unsigned;

namespace Domain {
const static char *ai_onnx      = "ai.onnx";
const static char *ai_graphcore = "ai.graphcore";
} // namespace Domain

// The Op identifier is defined by ONNX a tuple
// (https://github.com/onnx/onnx/blob/master/docs/Versioning.md)
// domain.type:version
struct OperatorIdentifier {
  OperatorIdentifier(const OpDomain &_domain,
                     const OpType &_type,
                     OpVersion _version)
      : domain(_domain), type(_type), version(_version) {

    // If no domain specified assume it is the default
    if (domain == "") {
      domain = Domain::ai_onnx;
    }
  }

  OpDomain domain;
  OpType type;
  OpVersion version;

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
  AiOnnxOperatorIdentifierV9(const OpType &_type)
      : OperatorIdentifier(Domain::ai_onnx, _type, 9) {}
};

struct AiGrapcoreOpIdV1 : public OperatorIdentifier {
  AiGrapcoreOpIdV1(const OpType &_type)
      : OperatorIdentifier(Domain::ai_graphcore, _type, 1) {}
};

std::ostream &operator<<(std::ostream &os, const OperatorIdentifier &opid);

// FFS There should be a way of generated this list of operators from the md
// files
namespace Onnx {

// When we have to support versioning suggest the following
// const static OperatorIdentifier Add_9(Domain::ai_onnx, "Add", 9);
// const static OperatorIdentifier Add = Add_9;

namespace Operators {
const static AiOnnxOperatorIdentifierV9 Abs("Abs");
const static AiOnnxOperatorIdentifierV9 Acos("Acos");
const static AiOnnxOperatorIdentifierV9 Acosh("Acosh");
const static AiOnnxOperatorIdentifierV9 Add("Add");

const static AiOnnxOperatorIdentifierV9 And("And");
const static AiOnnxOperatorIdentifierV9 ArgMax("ArgMax");
const static AiOnnxOperatorIdentifierV9 ArgMin("ArgMin");
const static AiOnnxOperatorIdentifierV9 Asin("Asin");
const static AiOnnxOperatorIdentifierV9 Asinh("Asinh");
const static AiOnnxOperatorIdentifierV9 Atan("Atan");
const static AiOnnxOperatorIdentifierV9 Atanh("Atanh");
const static AiOnnxOperatorIdentifierV9 AveragePool("AveragePool");
const static AiOnnxOperatorIdentifierV9
    BatchNormalization("BatchNormalization");
const static AiOnnxOperatorIdentifierV9 Cast("Cast");
const static AiOnnxOperatorIdentifierV9 Ceil("Ceil");
const static AiOnnxOperatorIdentifierV9 Clip("Clip");
const static AiOnnxOperatorIdentifierV9 Compress("Compress");
const static AiOnnxOperatorIdentifierV9 Concat("Concat");
const static AiOnnxOperatorIdentifierV9 Constant("Constant");
const static AiOnnxOperatorIdentifierV9 ConstantLike("ConstantLike");
const static AiOnnxOperatorIdentifierV9 Conv("Conv");
const static AiOnnxOperatorIdentifierV9 ConvTranspose("ConvTranspose");
const static AiOnnxOperatorIdentifierV9 Cos("Cos");
const static AiOnnxOperatorIdentifierV9 Cosh("Cosh");
const static AiOnnxOperatorIdentifierV9 DepthToSpace("DepthToSpace");
const static AiOnnxOperatorIdentifierV9 Div("Div");
const static AiOnnxOperatorIdentifierV9 Dropout("DropOut");
const static AiOnnxOperatorIdentifierV9 Elu("Elu");
const static AiOnnxOperatorIdentifierV9 Equal("Equal");
const static AiOnnxOperatorIdentifierV9 Erf("Erf");
const static AiOnnxOperatorIdentifierV9 Exp("Exp");
const static AiOnnxOperatorIdentifierV9 Expand("Expand");
const static AiOnnxOperatorIdentifierV9 EyeLike("EyeLike");
const static AiOnnxOperatorIdentifierV9 Flatten("Flatten");
const static AiOnnxOperatorIdentifierV9 Floor("Floor");
const static AiOnnxOperatorIdentifierV9 GRU("GRU");
const static AiOnnxOperatorIdentifierV9 Gather("Gather");
const static AiOnnxOperatorIdentifierV9 Gemm("Gemm");
const static AiOnnxOperatorIdentifierV9 GlobalAveragePool("GlobalAveragePool");
const static AiOnnxOperatorIdentifierV9 GlobalLpPool("GlobalLpPool");
const static AiOnnxOperatorIdentifierV9 GlobalMaxPool("GlobalMaxPool");
const static AiOnnxOperatorIdentifierV9 Greater("Greater");
const static AiOnnxOperatorIdentifierV9 HardSigmoid("HardSigmoid");
const static AiOnnxOperatorIdentifierV9 Hardmax("Hardmax");
const static AiOnnxOperatorIdentifierV9 Identity("Identity");
const static AiOnnxOperatorIdentifierV9 If("If");
const static AiOnnxOperatorIdentifierV9
    InstanceNormalization("InstanceNormalization");
const static AiOnnxOperatorIdentifierV9 IsNaN("IsNan");
const static AiOnnxOperatorIdentifierV9 LRN("LRN");
const static AiOnnxOperatorIdentifierV9 LSTM("LSTM");
const static AiOnnxOperatorIdentifierV9 LeakyRelu("LeakyRely");
const static AiOnnxOperatorIdentifierV9 Less("Less");
const static AiOnnxOperatorIdentifierV9 Log("Log");
const static AiOnnxOperatorIdentifierV9 LogSoftmax("LogSoftmax");
const static AiOnnxOperatorIdentifierV9 Loop("Loop");
const static AiOnnxOperatorIdentifierV9 LpNormalization("LpNormalization");
const static AiOnnxOperatorIdentifierV9 LpPool("LpPool");
const static AiOnnxOperatorIdentifierV9 MatMul("MatMul");
const static AiOnnxOperatorIdentifierV9 Max("Max");
const static AiOnnxOperatorIdentifierV9 MaxPool("MaxPool");
const static AiOnnxOperatorIdentifierV9 MaxRoiPool("MaxRoiPool");
const static AiOnnxOperatorIdentifierV9 MaxUnpool("MaxUnpool");
const static AiOnnxOperatorIdentifierV9 Mean("Mean");
const static AiOnnxOperatorIdentifierV9 Min("Min");
const static AiOnnxOperatorIdentifierV9 Mul("Mul");
const static AiOnnxOperatorIdentifierV9 Multinomial("Multinomial");
const static AiOnnxOperatorIdentifierV9 Neg("Neg");
const static AiOnnxOperatorIdentifierV9 Not("Not");
const static AiOnnxOperatorIdentifierV9 OneHot("OneHot");
const static AiOnnxOperatorIdentifierV9 Or("Or");
const static AiOnnxOperatorIdentifierV9 PRelu("PRelu");
const static AiOnnxOperatorIdentifierV9 Pad("Pad");
const static AiOnnxOperatorIdentifierV9 Pow("Pow");
const static AiOnnxOperatorIdentifierV9 RNN("RNN");
const static AiOnnxOperatorIdentifierV9 RandomNormal("RandomNormal");
const static AiOnnxOperatorIdentifierV9 RandomNormalLike("RandomNormalLike");
const static AiOnnxOperatorIdentifierV9 RandomUniform("RandomUniform");
const static AiOnnxOperatorIdentifierV9 RandomUniformLike("RandomUniformLike");
const static AiOnnxOperatorIdentifierV9 Reciprocal("Reciprocal");
const static AiOnnxOperatorIdentifierV9 ReduceL1("ReduceL1");
const static AiOnnxOperatorIdentifierV9 ReduceL2("ReduceL2");
const static AiOnnxOperatorIdentifierV9 ReduceLogSum("ReduceLogSum");
const static AiOnnxOperatorIdentifierV9 ReduceLogSumExp("ReduceLogSumExp");
const static AiOnnxOperatorIdentifierV9 ReduceMax("ReduceMax");
const static AiOnnxOperatorIdentifierV9 ReduceMean("ReduceMean");
const static AiOnnxOperatorIdentifierV9 ReduceMin("ReduceMin");
const static AiOnnxOperatorIdentifierV9 ReduceProd("ReduceProd");
const static AiOnnxOperatorIdentifierV9 ReduceSum("ReduceSum");
const static AiOnnxOperatorIdentifierV9 ReduceSumSquare("ReduceSumSquare");
const static AiOnnxOperatorIdentifierV9 Relu("Relu");
const static AiOnnxOperatorIdentifierV9 Reshape("Reshape");
const static AiOnnxOperatorIdentifierV9 Scan("Scan");
const static AiOnnxOperatorIdentifierV9 Scatter("Scatter");
const static AiOnnxOperatorIdentifierV9 Selu("Selu");
const static AiOnnxOperatorIdentifierV9 Shape("Shape");
const static AiOnnxOperatorIdentifierV9 Sigmoid("Sigmoid");
const static AiOnnxOperatorIdentifierV9 Sign("Sign");
const static AiOnnxOperatorIdentifierV9 Sin("Sin");
const static AiOnnxOperatorIdentifierV9 Sinh("Sinh");
const static AiOnnxOperatorIdentifierV9 Size("Size");
const static AiOnnxOperatorIdentifierV9 Slice("Slice");
const static AiOnnxOperatorIdentifierV9 Softmax("Softmax");
const static AiOnnxOperatorIdentifierV9 Softplus("Softplus");
const static AiOnnxOperatorIdentifierV9 Softsign("Softsign");
const static AiOnnxOperatorIdentifierV9 SpaceToDepth("SpaceToDepth");
const static AiOnnxOperatorIdentifierV9 Split("Split");
const static AiOnnxOperatorIdentifierV9 Sqrt("Sqrt");
const static AiOnnxOperatorIdentifierV9 Squeeze("Squeeze");
const static AiOnnxOperatorIdentifierV9 Sub("Sub");
const static AiOnnxOperatorIdentifierV9 Sum("Sum");
const static AiOnnxOperatorIdentifierV9 Tan("Tan");
const static AiOnnxOperatorIdentifierV9 Tanh("Tanh");
const static AiOnnxOperatorIdentifierV9 Tile("Tile");
const static AiOnnxOperatorIdentifierV9 TopK("TopK");
const static AiOnnxOperatorIdentifierV9 Transpose("Transpose");
const static AiOnnxOperatorIdentifierV9 Unsqueeze("Unsqueeze");
const static AiOnnxOperatorIdentifierV9 Upsample("Upsample");
const static AiOnnxOperatorIdentifierV9 Xor("Xor");
// experimental
const static AiOnnxOperatorIdentifierV9 ATen("ATen");
const static AiOnnxOperatorIdentifierV9 Affine("Affine");
const static AiOnnxOperatorIdentifierV9 ConstantFill("ConstantFill");
const static AiOnnxOperatorIdentifierV9 Crop("Crop");
const static AiOnnxOperatorIdentifierV9 DynamicSlice("DynamicSlice");
const static AiOnnxOperatorIdentifierV9 GRUUnit("GRUUnit");
const static AiOnnxOperatorIdentifierV9 GivenTensorFill("GivenTensorFill");
const static AiOnnxOperatorIdentifierV9 ImageScaler("ImageScaler");
const static AiOnnxOperatorIdentifierV9
    ParametricSoftplus("ParametricSoftplus");
const static AiOnnxOperatorIdentifierV9 Scale("Scale");
const static AiOnnxOperatorIdentifierV9 ScaledTanh("ScaledTanh");
const static AiOnnxOperatorIdentifierV9 ThresholdedRelu("ThresholdedRelu");
} // namespace Operators

namespace GradOperators {
// constexpr static char Abs[] = "Abs";
// constexpr static char Acos[] = "Acos";
// constexpr static char Acosh[] = "Acosh";
const static AiGrapcoreOpIdV1 AddArg0Grad("AddArg0Grad");
const static AiGrapcoreOpIdV1 AddArg1Grad("AddArg1Grad");
// constexpr static char And[] = "And";
// constexpr static char ArgMax[] = "ArgMax";
// constexpr static char ArgMin[] = "ArgMin";
// constexpr static char Asin[] = "Asin";
// constexpr static char Asinh[] = "Asinh";
// constexpr static char Atan[] = "Atan";
// constexpr static char Atanh[] = "Atanh";
const static AiGrapcoreOpIdV1 AveragePoolGrad("AveragePoolGrad");
const static AiGrapcoreOpIdV1 BatchNormalizationGrad("BatchNormalizationGrad");
// constexpr static char Cast[] = "Cast";
// constexpr static char Ceil[] = "Ceil";
// constexpr static char Clip[] = "Clip";
// constexpr static char Compress[] = "Compress";
// constexpr static char Concat[] = "Concat";
// constexpr static char Constant[] = "Constant";
// constexpr static char ConstantLike[] = "ConstantLike";
const static AiGrapcoreOpIdV1 ConvDataGrad("ConvDataGrad");
const static AiGrapcoreOpIdV1 ConvWeightsGrad("ConvWeightsGrad");
// constexpr static char ConvTranspose[] = "ConvTranspose";
const static AiGrapcoreOpIdV1 CosGrad("CosGrad");
// constexpr static char Cosh[] = "Cosh"
// constexpr static char DepthToSpace[] = "DepthToSpace";
const static AiGrapcoreOpIdV1 DivArg0Grad("DivArg0Grad");
const static AiGrapcoreOpIdV1 DivArg1Grad("DivArg1Grad");
// constexpr static char Dropout[] = "DropOut";
// constexpr static char Elu[] = "Elu";
// constexpr static char Equal[] = "Equal";
// constexpr static char Erf[] = "Erf";
const static AiGrapcoreOpIdV1 ExpGrad("ExpGrad");
// constexpr static char Expand[] = "Expand";
// constexpr static char EyeLike[] = "EyeLike";
// constexpr static char Flatten[] = "Flatten";
// constexpr static char Floor[] = "Floor";
// constexpr static char GRU[] = "GRU";
// constexpr static char Gather[] = "Gather";
// constexpr static char Gemm[] = "Gemm";
// constexpr static char GlobalAveragePool[] = "GlobalAveragePool";
// constexpr static char GlobalLpPool[] = "GlobalLpPool";
// constexpr static char GlobalMaxPool[] = "GlobalMaxPool";
// constexpr static char Greater[] = "Greater";
// constexpr static char HardSigmoid[] = "HardSigmoid";
// constexpr static char Hardmax[] = "Hardmax";
const static AiGrapcoreOpIdV1 IdentityGrad("IdentityGrad");
// constexpr static char If[] = "If";
// constexpr static char InstanceNormalization[] = "InstanceNormalization";
// constexpr static char IsNaN[] = "IsNan";
// constexpr static char LRN[] = "LRN";
// constexpr static char LSTM[] = "LSTM";
// constexpr static char LeakyRelu[] = "LeakyRely";
// constexpr static char Less[] = "Less";
// constexpr static char Log[] = "Log";
// constexpr static char LogSoftmax[] = "LogSoftmax";
// constexpr static char Loop[] = "Loop";
// constexpr static char LpNormalization[] = "LpNormalization";
// constexpr static char LpPool[] = "LpPool";
const static AiGrapcoreOpIdV1 MatMulLhsGrad("MatMulLhsGrad");
const static AiGrapcoreOpIdV1 MatMulRhsGrad("MatMulRhsGrad");
// constexpr static char Max[] = "Max";
const static AiGrapcoreOpIdV1 MaxPoolGrad("MaxPoolGrad");
// constexpr static char MaxRoiPool[] = "MaxRoiPooL";
// constexpr static char MaxUnpool[] = "MaxUnpool";
// constexpr static char Mean[] = "Mean";
// constexpr static char Min[] = "Min";
const static AiGrapcoreOpIdV1 MulArg0Grad("MulArg0Grad");
const static AiGrapcoreOpIdV1 MulArg1Grad("MulArg1Grad");
// constexpr static char Multinomial[] = "Multinomial";
const static AiGrapcoreOpIdV1 NegGrad("NegGrad");
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
const static AiGrapcoreOpIdV1 ReciprocalGrad("ReciprocalGrad");
// constexpr static char ReduceL1[] = "ReduceL1";
// constexpr static char ReduceL2[] = "ReduceL2";
// constexpr static char ReduceLogSum[] = "ReduceLogSum";
// constexpr static char ReduceLogSumExp[] = "ReduceLogSumExp";
// constexpr static char ReduceMax[] = "ReduceMax";
// constexpr static char ReduceMean[] = "ReduceMean";
// constexpr static char ReduceMin[] = "ReduceMin";
// constexpr static char ReduceProd[] = "ReduceProd";
const static AiGrapcoreOpIdV1 ReduceSumGrad("ReduceSumGrad");
// constexpr static char ReduceSumSquare[] = "ReduceSumSquare";
const static AiGrapcoreOpIdV1 ReluGrad("ReluGrad");
const static AiGrapcoreOpIdV1 ReshapeGrad("ReshapeGrad");
// constexpr static char Scan[] = "Scan";
// constexpr static char Scatter[] = "Scatter";
// constexpr static char Selu[] = "Selu";
// constexpr static char Shape[] = "Shape";
const static AiGrapcoreOpIdV1 SigmoidGrad("SigmoidGrad");
// constexpr static char Sign[] = "Sign";
const static AiGrapcoreOpIdV1 SinGrad("SinGrad");
// constexpr static char Sinh[] = "Sinh";
// constexpr static char Size[] = "Size";
// constexpr static char Slice[] = "Slice";
const static AiGrapcoreOpIdV1 SoftmaxGrad("SoftmaxGrad");
// constexpr static char Softplus[] = "Softplus";
// constexpr static char Softsign[] = "Softsign";
// constexpr static char SpaceToDepth[] = "SpaceToDepth";
// constexpr static char Split[] = "Split";
const static AiGrapcoreOpIdV1 SqrtGrad("SqrtGrad");
const static AiGrapcoreOpIdV1 SqueezeGrad("SqueezeGrad");
const static AiGrapcoreOpIdV1 SubArg0Grad("SubArg0Grad");
const static AiGrapcoreOpIdV1 SubArg1Grad("SubArg1Grad");
// constexpr static char Sum[] = "Sum";
// constexpr static char Tan[] = "Tan";
const static AiGrapcoreOpIdV1 TanhGrad("TanhGrad");
// constexpr static char Tile[] = "Tile";
// constexpr static char TopK[] = "TopK";
const static AiGrapcoreOpIdV1 TransposeGrad("TransposeGrad");
// constexpr static char Unsqueeze[] = "Unsqueeze";
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
const static AiGrapcoreOpIdV1 ScaleGrad("ScaleGrad");
// constexpr static char ScaledTanh[] = "ScaledTanh";
// constexpr static char ThresholdedRelu[] = "ThresholdedRelu";

} // namespace GradOperators

namespace CustomOperators {
const static AiGrapcoreOpIdV1 AddBias("AddBias");
const static AiGrapcoreOpIdV1 Subsample("Subsample");
const static AiGrapcoreOpIdV1 Square("Square");

const static AiGrapcoreOpIdV1 ReluInplace("ReluInplace");
const static AiGrapcoreOpIdV1 L1("L1");
const static AiGrapcoreOpIdV1 Nll("Nll");

const static AiGrapcoreOpIdV1 SgdVarUpdate("SGDVarUpdate");
const static AiGrapcoreOpIdV1 ConstSgdVarUpdate("ConstSGDVarUpdate");
} // namespace CustomOperators

namespace CustomGradOperators {
const static AiGrapcoreOpIdV1 AddBiasBiasGrad("AddBiasBiasGrad");
const static AiGrapcoreOpIdV1 AddBiasDataGrad("AddBiasDataGrad");
const static AiGrapcoreOpIdV1 SoftmaxGradDirect("SoftmaxGradDirect");
const static AiGrapcoreOpIdV1 SubsampleGrad("SubsampleGrad");
const static AiGrapcoreOpIdV1 L1Grad("L1Grad");
const static AiGrapcoreOpIdV1 NllGrad("NllGrad");
} // namespace CustomGradOperators
} // namespace Onnx

} // namespace poponnx

#endif
