// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPIDENTIFIER_HPP
#define GUARD_NEURALNET_OPIDENTIFIER_HPP

#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>

namespace popart {

namespace Domain {
constexpr const char *ai_onnx      = "ai.onnx";
constexpr const char *ai_onnx_ml   = "ai.onnx.ml";
constexpr const char *ai_graphcore = "ai.graphcore";
} // namespace Domain

// Default opset versions for domains
const static int64_t defaultAiOnnxOpset = 10;

// T12084
// const static int64_t defaultAiOnnxOpset      = 11;

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

  bool operator<(const OperatorIdentifier &rhs) const {
    return domain < rhs.domain || (domain == rhs.domain && type < rhs.type) ||
           (domain == rhs.domain && type == rhs.type && version < rhs.version);
  }
};

// The following does not work as we are in the popart namesapace >>
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
}
} // namespace Onnx

#include <popart/opidentifier.hpp.gen>

namespace Onnx {
namespace GradOperators {
const static AiGraphcoreOpIdV1 AbsGrad("AbsGrad");
const static AiGraphcoreOpIdV1 AddArg0Grad("AddArg0Grad");
const static AiGraphcoreOpIdV1 AddArg1Grad("AddArg1Grad");
const static AiGraphcoreOpIdV1 AsinGrad("AsinGrad");
const static AiGraphcoreOpIdV1 Atan2Arg0Grad("Atan2Arg0Grad");
const static AiGraphcoreOpIdV1 Atan2Arg1Grad("Atan2Arg1Grad");
const static AiGraphcoreOpIdV1 AtanGrad("AtanGrad");
const static AiGraphcoreOpIdV1 AutoLossScaleProxyGrad("AutoLossScaleProxyGrad");
const static AiGraphcoreOpIdV1 AveragePoolGrad("AveragePoolGrad");
const static AiGraphcoreOpIdV1 BatchNormalizationGrad("BatchNormalizationGrad");
const static AiGraphcoreOpIdV1 CastGrad("CastGrad");
const static AiGraphcoreOpIdV1 ClipGrad("ClipGrad");
const static AiGraphcoreOpIdV1 ConcatGrad("ConcatGrad");
const static AiGraphcoreOpIdV1 ConvDataGrad("ConvDataGrad");
const static AiGraphcoreOpIdV1 ConvWeightsGrad("ConvWeightsGrad");
const static AiGraphcoreOpIdV1 CosGrad("CosGrad");
const static AiGraphcoreOpIdV1 CumSumGrad("CumSumGrad");
const static AiGraphcoreOpIdV1 DivArg0Grad("DivArg0Grad");
const static AiGraphcoreOpIdV1 DivArg1Grad("DivArg1Grad");
const static AiGraphcoreOpIdV1 DropoutGrad("DropoutGrad");
const static AiGraphcoreOpIdV1 EluGrad("EluGrad");
const static AiGraphcoreOpIdV1 ErfGrad("ErfGrad");
const static AiGraphcoreOpIdV1 ExpandGrad("ExpandGrad");
const static AiGraphcoreOpIdV1 ExpGrad("ExpGrad");
const static AiGraphcoreOpIdV1 Expm1Grad("Expm1Grad");
const static AiGraphcoreOpIdV1 GatherGrad("GatherGrad");
const static AiGraphcoreOpIdV1 GeluGrad("GeluGrad");
const static AiGraphcoreOpIdV1 GlobalAveragePoolGrad("GlobalAveragePoolGrad");
const static AiGraphcoreOpIdV1 GlobalMaxPoolGrad("GlobalMaxPoolGrad");
const static AiGraphcoreOpIdV1 GroupNormalizationGrad("GroupNormalizationGrad");
const static AiGraphcoreOpIdV1 GRUGrad("GRUGrad");
const static AiGraphcoreOpIdV1 HardSigmoidGrad("HardSigmoidGrad");
const static AiGraphcoreOpIdV1 IdentityGrad("IdentityGrad");
const static AiGraphcoreOpIdV1 IdentityLossGrad("IdentityLossGrad");
const static AiGraphcoreOpIdV1
    InstanceNormalizationGrad("InstanceNormalizationGrad");
const static AiGraphcoreOpIdV1 LeakyReluGrad("LeakyReluGrad");
const static AiGraphcoreOpIdV1 Log1pGrad("Log1pGrad");
const static AiGraphcoreOpIdV1 LogGrad("LogGrad");
const static AiGraphcoreOpIdV1 LogSoftmaxGrad("LogSoftmaxGrad");
const static AiGraphcoreOpIdV1 LRN("LRN");
const static AiGraphcoreOpIdV1 LRNGrad("LRNGrad");
const static AiGraphcoreOpIdV1 LSTMGrad("LSTMGrad");
const static AiGraphcoreOpIdV1 MatMulLhsGrad("MatMulLhsGrad");
const static AiGraphcoreOpIdV1 MatMulRhsGrad("MatMulRhsGrad");
const static AiGraphcoreOpIdV1 MaxArgGrad("MaxArgGrad");
const static AiGraphcoreOpIdV1 MaxPoolGrad("MaxPoolGrad");
const static AiGraphcoreOpIdV1 MeanArgGrad("MeanArgGrad");
const static AiGraphcoreOpIdV1 MinArgGrad("MinArgGrad");
const static AiGraphcoreOpIdV1 FmodArg0Grad("FmodArg0Grad");
const static AiGraphcoreOpIdV1 MulArg0Grad("MulArg0Grad");
const static AiGraphcoreOpIdV1 MulArg1Grad("MulArg1Grad");
const static AiGraphcoreOpIdV1 MultiConvDataGrad("MultiConvDataGrad");
const static AiGraphcoreOpIdV1 MultiConvWeightsGrad("MultiConvWeightsGrad");
const static AiGraphcoreOpIdV1 NegGrad("NegGrad");
const static AiGraphcoreOpIdV1 OneHotGrad("OneHotGrad");
const static AiGraphcoreOpIdV1 PadGrad("PadGrad");
const static AiGraphcoreOpIdV1 PopartLSTMGrad("PopartLSTMGrad");
const static AiGraphcoreOpIdV1 PowArg0Grad("PowArg0Grad");
const static AiGraphcoreOpIdV1 PowArg1Grad("PowArg1Grad");
const static AiGraphcoreOpIdV1 ReciprocalGrad("ReciprocalGrad");
const static AiGraphcoreOpIdV1 ReduceL1Grad("ReduceL1Grad");
const static AiGraphcoreOpIdV1 ReduceL2Grad("ReduceL2Grad");
const static AiGraphcoreOpIdV1 ReduceLogSumExpGrad("ReduceLogSumExpGrad");
const static AiGraphcoreOpIdV1 ReduceLogSumGrad("ReduceLogSumGrad");
const static AiGraphcoreOpIdV1 ReduceMaxGrad("ReduceMaxGrad");
const static AiGraphcoreOpIdV1 ReduceMeanGrad("ReduceMeanGrad");
const static AiGraphcoreOpIdV1 ReduceMedianGrad("ReduceMedianGrad");
const static AiGraphcoreOpIdV1 ReduceMinGrad("ReduceMinGrad");
const static AiGraphcoreOpIdV1 ReduceProdGrad("ReduceProdGrad");
const static AiGraphcoreOpIdV1 ReduceSumGrad("ReduceSumGrad");
const static AiGraphcoreOpIdV1 ReduceSumSquareGrad("ReduceSumSquareGrad");
const static AiGraphcoreOpIdV1 ReluGrad("ReluGrad");
const static AiGraphcoreOpIdV1 ReshapeGrad("ReshapeGrad");
const static AiGraphcoreOpIdV1 ResizeGrad("ResizeGrad");
const static AiGraphcoreOpIdV1 ReverseGrad("ReverseGrad");
const static AiGraphcoreOpIdV1 Scale("Scale");
const static AiGraphcoreOpIdV1 ScaleGrad("ScaleGrad");
const static AiGraphcoreOpIdV1 ScatterDataGrad("ScatterDataGrad");
const static AiGraphcoreOpIdV1 ScatterUpdateGrad("ScatterUpdateGrad");
const static AiGraphcoreOpIdV1 SeluGrad("SeluGrad");
const static AiGraphcoreOpIdV1 ShrinkGrad("ShrinkGrad");
const static AiGraphcoreOpIdV1 SigmoidGrad("SigmoidGrad");
const static AiGraphcoreOpIdV1 SignGrad("SignGrad");
const static AiGraphcoreOpIdV1 SinGrad("SinGrad");
const static AiGraphcoreOpIdV1 SinhGrad("SinhGrad");
const static AiGraphcoreOpIdV1 SliceGrad("SliceGrad");
const static AiGraphcoreOpIdV1 SoftmaxGrad("SoftmaxGrad");
const static AiGraphcoreOpIdV1 SoftPlusGrad("SoftPlusGrad");
const static AiGraphcoreOpIdV1 SoftSignGrad("SoftSignGrad");
const static AiGraphcoreOpIdV1 SplitGrad("SplitGrad");
const static AiGraphcoreOpIdV1 SqrtGrad("SqrtGrad");
const static AiGraphcoreOpIdV1 SubArg0Grad("SubArg0Grad");
const static AiGraphcoreOpIdV1 SubArg1Grad("SubArg1Grad");
const static AiGraphcoreOpIdV1 SumArgGrad("SumArgGrad");
const static AiGraphcoreOpIdV1 TanhGrad("TanhGrad");
const static AiGraphcoreOpIdV1 ThresholdedReluGrad("ThresholdedReluGrad");
const static AiGraphcoreOpIdV1 TileGrad("TileGrad");
const static AiGraphcoreOpIdV1 TopKGrad("TopKGrad");
const static AiGraphcoreOpIdV1 TransposeGrad("TransposeGrad");
const static AiGraphcoreOpIdV1 WhereXGrad("WhereXGrad");
const static AiGraphcoreOpIdV1 WhereYGrad("WhereYGrad");
} // namespace GradOperators

namespace CustomOperators {
const static AiGraphcoreOpIdV1 Accumulate("Accumulate");
const static AiGraphcoreOpIdV1 AccumulatorScale("AccumulatorScale");
const static AiGraphcoreOpIdV1 AdaDeltaUpdater("AdaDeltaUpdater");
const static AiGraphcoreOpIdV1 AdamCombo("AdamCombo");
const static AiGraphcoreOpIdV1 AdaptiveCombo("AdaptiveCombo");
const static AiGraphcoreOpIdV1 RescaleAccumulate("RescaleAccumulate");
const static AiGraphcoreOpIdV1 AdamUpdater("AdamUpdater");
const static AiGraphcoreOpIdV1 AdamVarUpdate("AdamVarUpdate");
const static AiGraphcoreOpIdV1 AddBias("AddBias");
const static AiGraphcoreOpIdV1 AddBiasInplace("AddBiasInplace");
const static AiGraphcoreOpIdV1 AddLhsInplace("AddLhsInplace");
const static AiGraphcoreOpIdV1 AddRhsInplace("AddRhsInplace");
const static AiGraphcoreOpIdV1 AsinInplace("AsinInplace");
const static AiGraphcoreOpIdV1 Atan2_1("Atan2", 2, 1);
const static AiGraphcoreOpIdV1 Atan2Inplace("Atan2Inplace");
const static AiGraphcoreOpIdV1 AtanInplace("AtanInplace");
const static AiGraphcoreOpIdV1 AutoLossScaleProxy("AutoLossScaleProxy", 1, 1);
const static AiGraphcoreOpIdV1 Call_1("Call");
const static AiGraphcoreOpIdV1 CeilInplace("CeilInplace");
const static AiGraphcoreOpIdV1 ClipInplace("ClipInplace");
const static AiGraphcoreOpIdV1 ConcatGradInplace("ConcatGradInplace");
const static AiGraphcoreOpIdV1 ConcatInplace("ConcatInplace");
const static AiGraphcoreOpIdV1 ConvFlipWeights("ConvFlipWeights");
const static AiGraphcoreOpIdV1 CopyVarUpdate("CopyVarUpdate");
const static AiGraphcoreOpIdV1 Ctc("Ctc", 4, 2);
const static AiGraphcoreOpIdV1
    CtcBeamSearchDecoder("CtcBeamSearchDecoder", 2, 3);
const static AiGraphcoreOpIdV1 DepthToSpace("DepthToSpace", 1, 1);
const static AiGraphcoreOpIdV1 Detach_1("Detach", 1, 1);
const static AiGraphcoreOpIdV1 DetachInplace("DetachInplace");
const static AiGraphcoreOpIdV1 DynamicAdd_1("DynamicAdd", 3, 1);
const static AiGraphcoreOpIdV1 DynamicAddInplace("DynamicAddInplace", 3, 1);
const static AiGraphcoreOpIdV1 DynamicSlice_1("DynamicSlice", 2, 1);
const static AiGraphcoreOpIdV1 SequenceSlice_1("SequenceSlice", 5, 1);
const static AiGraphcoreOpIdV1
    SequenceSliceInplace("SequenceSliceInplace", 5, 1);
const static AiGraphcoreOpIdV1 DynamicUpdate_1("DynamicUpdate", 3, 1);
const static AiGraphcoreOpIdV1
    DynamicUpdateInplace("DynamicUpdateInplace", 3, 1);
const static AiGraphcoreOpIdV1 DynamicZero_1("DynamicZero", 2, 1);
const static AiGraphcoreOpIdV1 DynamicZeroInplace("DynamicZeroInplace", 2, 1);
const static AiGraphcoreOpIdV1 EluInplace("EluInplace");
const static AiGraphcoreOpIdV1 ExpandInplace("ExpandInplace");
const static AiGraphcoreOpIdV1 ExpInplace("ExpInplace");
const static AiGraphcoreOpIdV1 Expm1_1("Expm1", 1, 1);
const static AiGraphcoreOpIdV1 Expm1Inplace("Expm1Inplace");
const static AiGraphcoreOpIdV1 FlattenInplace("FlattenInplace");
const static AiGraphcoreOpIdV1 FloorInplace("FloorInplace");
const static AiGraphcoreOpIdV1 Fmod("Fmod", 2, 1);
const static AiGraphcoreOpIdV1 Gelu_1("Gelu", 1, 1);
const static AiGraphcoreOpIdV1 GeluInplace("GeluInplace");
const static AiGraphcoreOpIdV1 GetRandomSeed("GetRandomSeed");
const static AiGraphcoreOpIdV1 GradCopyFromHost("GradCopyFromHost");
const static AiGraphcoreOpIdV1 GradCopyToHost("GradCopyToHost");
const static AiGraphcoreOpIdV1 GradientAccumulation("GradientAccl");
const static AiGraphcoreOpIdV1 GroupNormalization_1("GroupNormalization", 3, 3);
const static AiGraphcoreOpIdV1 HardSigmoidInplace("HardSigmoidInplace");
const static AiGraphcoreOpIdV1 Histogram("Histogram", 1, 1);
const static AiGraphcoreOpIdV1 HostSGD0VarUpdate("HostSGD0VarUpdate");
const static AiGraphcoreOpIdV1 IdentityInplace("IdentityInplace");
const static AiGraphcoreOpIdV1 IdentityLoss("IdentityLoss", 1, 1);
const static AiGraphcoreOpIdV1 Init_1("Init", 0, 1);
const static AiGraphcoreOpIdV1 IoTileCopy("IoTileCopy");
const static AiGraphcoreOpIdV1 IpuCopy("IpuCopy");
const static AiGraphcoreOpIdV1 L1("L1", 1, 1);
const static AiGraphcoreOpIdV1 LambSquare("LambSquare");
const static AiGraphcoreOpIdV1 LeakyReluInplace("LeakyReluInplace");
const static AiGraphcoreOpIdV1 Log1p_1("Log1p", 1, 1);
const static AiGraphcoreOpIdV1 Log1pInplace("Log1pInplace");
const static AiGraphcoreOpIdV1 LogSoftmaxInplace("LogSoftmaxInplace");
const static AiGraphcoreOpIdV1 LossScaleUpdate("LossScaleUpdate");
const static AiGraphcoreOpIdV1 LSTM_1("LSTM", 4, 2);
const static AiGraphcoreOpIdV1 ModifyRandomSeed("ModifyRandomSeed");
const static AiGraphcoreOpIdV1 MulLhsInplace("MulLhsInplace");
const static AiGraphcoreOpIdV1 MulRhsInplace("MulRhsInplace");
const static AiGraphcoreOpIdV1 MultiConv_1("MultiConv");
const static AiGraphcoreOpIdV1 Nll("Nll", 2, 1);
const static AiGraphcoreOpIdV1 Nop_1("Nop", 1, 1);
const static AiGraphcoreOpIdV1 PadInplace("PadInplace");
const static AiGraphcoreOpIdV1 PowLhsInplace("PowLhsInplace");
const static AiGraphcoreOpIdV1 PrintTensor_1("PrintTensor", 1, 1);
const static AiGraphcoreOpIdV1 ReduceMedian("ReduceMedian", 1, 2);
const static AiGraphcoreOpIdV1 PackedDataBlock("PackedDataBlock", {}, 1);
const static AiGraphcoreOpIdV1 ReluInplace("ReluInplace");
const static AiGraphcoreOpIdV1 Remainder("Remainder", 2, 1);
const static AiGraphcoreOpIdV1 MultiExchange("MultiExchange");
const static AiGraphcoreOpIdV1 RemoteLoad("RemoteLoad", {1, 2}, 1);
const static AiGraphcoreOpIdV1 RemoteStore("RemoteStore", {1, 2}, 0);
const static AiGraphcoreOpIdV1 HostLoad("HostLoad", {1, 1}, 0);
const static AiGraphcoreOpIdV1 HostStore("HostStore", {1, 0}, 0);
const static AiGraphcoreOpIdV1 ReplicatedAllGather("ReplicatedAllGather");
const static AiGraphcoreOpIdV1 ReplicatedAllReduce("ReplicatedAllReduce", 1, 1);
const static AiGraphcoreOpIdV1
    ReplicatedAllReduceInplace("ReplicatedAllReduceInplace", 1, 1);
const static AiGraphcoreOpIdV1
    ReplicatedReduceScatter("ReplicatedReduceScatter");
const static AiGraphcoreOpIdV1 ResetAccumulation("ResetAccl");
const static AiGraphcoreOpIdV1 ReshapeInplace("ReshapeInplace");
const static AiGraphcoreOpIdV1 Reshape_1("Reshape", 1, 1);
const static AiGraphcoreOpIdV1 Resize("Resize");
const static AiGraphcoreOpIdV1 Restore("Restore");
const static AiGraphcoreOpIdV1 RestoreInplace("RestoreInplace");
const static AiGraphcoreOpIdV1 Reverse("Reverse", 1, 1);
const static AiGraphcoreOpIdV1 ReverseInplace("ReverseInplace");
const static AiGraphcoreOpIdV1 RMSPropUpdater("RMSPropUpdater");
const static AiGraphcoreOpIdV1 Round_1("Round", 1, 1);
const static AiGraphcoreOpIdV1 RoundInplace("RoundInplace");
const static AiGraphcoreOpIdV1 Scale_1("Scale", 1, 1);
const static AiGraphcoreOpIdV1 BinaryConstScalar("BinaryConstScalar", 1, 1);
const static AiGraphcoreOpIdV1 ScaledAdd("ScaledAdd", {2, 4}, 1);
const static AiGraphcoreOpIdV1 ScaledAddLhsInplace("ScaledAddLhsInplace");
const static AiGraphcoreOpIdV1 ScaledAddRhsInplace("ScaledAddRhsInplace");
const static AiGraphcoreOpIdV1 ScaledVarUpdate("ScaledVarUpdate");
const static AiGraphcoreOpIdV1 ScaleInplace("ScaleInplace");
const static AiGraphcoreOpIdV1 ScatterReduce("ScatterReduce", 2, 1);
const static AiGraphcoreOpIdV1 SeluInplace("SeluInplace");
const static AiGraphcoreOpIdV1 SGD0Combo("SGD0Combo");
const static AiGraphcoreOpIdV1 SGD0VarUpdate("SGD0VarUpdate");
const static AiGraphcoreOpIdV1 SGD1AcclUpdate("SGD1AcclUpdate");
const static AiGraphcoreOpIdV1 SGD1Combo("SGD1Combo");
const static AiGraphcoreOpIdV1 SGD1VarUpdate("SGD1VarUpdate");
const static AiGraphcoreOpIdV1 SGD2Combo("SGD2Combo");
const static AiGraphcoreOpIdV1 ShapedDropout_1("ShapedDropout", 1, 1);
const static AiGraphcoreOpIdV1 ShrinkInplace("ShrinkInplace");
const static AiGraphcoreOpIdV1 SigmoidInplace("SigmoidInplace");
const static AiGraphcoreOpIdV1 SignInplace("SignInplace");
const static AiGraphcoreOpIdV1 SinhInplace("SinhInplace");
const static AiGraphcoreOpIdV1 SliceInplace("SliceInplace");
const static AiGraphcoreOpIdV1 SoftmaxInplace("SoftmaxInplace");
const static AiGraphcoreOpIdV1 SoftPlusInplace("SoftPlusInplace");
const static AiGraphcoreOpIdV1 SoftSignInplace("SoftSignInplace");
const static AiGraphcoreOpIdV1 Square("Square");
const static AiGraphcoreOpIdV1 SqueezeInplace("SqueezeInplace");
const static AiGraphcoreOpIdV1 Stash("Stash");
const static AiGraphcoreOpIdV1 Subsample_1("Subsample", 1, 1);
const static AiGraphcoreOpIdV1 SubsampleInplace("SubsampleInplace");
const static AiGraphcoreOpIdV1 Sync("Sync");
const static AiGraphcoreOpIdV1 ThresholdedReluInplace("ThresholdedReluInplace");
const static AiGraphcoreOpIdV1 TransposeInplace("TransposeInplace");
const static AiGraphcoreOpIdV1 Zeros_1("Zeros");
const static AiGraphcoreOpIdV1 ZerosLike_1("ZerosLike");
const static AiGraphcoreOpIdV1 Abort("Abort");
const static AiGraphcoreOpIdV1 BitwiseAnd("BitwiseAnd", 2, 1);
const static AiGraphcoreOpIdV1 BitwiseNot("BitwiseNot", 1, 1);
const static AiGraphcoreOpIdV1 BitwiseOr("BitwiseOr", 2, 1);
const static AiGraphcoreOpIdV1 BitwiseXor("BitwiseXor", 2, 1);
const static AiGraphcoreOpIdV1 BitwiseXnor("BitwiseXnor", 2, 1);
} // namespace CustomOperators

namespace AiGraphcore {
namespace OpSet1 {
const static OperatorIdentifier Atan2 = CustomOperators::Atan2_1;
const static OperatorIdentifier AutoLossScaleProxy =
    CustomOperators::AutoLossScaleProxy;
const static OperatorIdentifier BitwiseAnd  = CustomOperators::BitwiseAnd;
const static OperatorIdentifier BitwiseNot  = CustomOperators::BitwiseNot;
const static OperatorIdentifier BitwiseOr   = CustomOperators::BitwiseOr;
const static OperatorIdentifier BitwiseXor  = CustomOperators::BitwiseXor;
const static OperatorIdentifier BitwiseXnor = CustomOperators::BitwiseXnor;
const static OperatorIdentifier Call        = CustomOperators::Call_1;
const static OperatorIdentifier Ctc         = CustomOperators::Ctc;
const static OperatorIdentifier CtcBeamSearchDecoder =
    CustomOperators::CtcBeamSearchDecoder;
const static OperatorIdentifier DepthToSpace = CustomOperators::DepthToSpace;
const static OperatorIdentifier Detach       = CustomOperators::Detach_1;
const static OperatorIdentifier DynamicAdd   = CustomOperators::DynamicAdd_1;
const static OperatorIdentifier DynamicSlice = CustomOperators::DynamicSlice_1;
const static OperatorIdentifier SequenceSlice =
    CustomOperators::SequenceSlice_1;
const static OperatorIdentifier DynamicUpdate =
    CustomOperators::DynamicUpdate_1;
const static OperatorIdentifier DynamicZero = CustomOperators::DynamicZero_1;
const static OperatorIdentifier Expm1       = CustomOperators::Expm1_1;
const static OperatorIdentifier Fmod        = CustomOperators::Fmod;
const static OperatorIdentifier Gelu        = CustomOperators::Gelu_1;
const static OperatorIdentifier GroupNormalization =
    CustomOperators::GroupNormalization_1;
const static OperatorIdentifier IdentityLoss = CustomOperators::IdentityLoss;
const static OperatorIdentifier Init         = CustomOperators::Init_1;
const static OperatorIdentifier L1           = CustomOperators::L1;
const static OperatorIdentifier Log1p        = CustomOperators::Log1p_1;
const static OperatorIdentifier LSTM         = CustomOperators::LSTM_1;
const static OperatorIdentifier MultiConv    = CustomOperators::MultiConv_1;
const static OperatorIdentifier Nll          = CustomOperators::Nll;
const static OperatorIdentifier Nop          = CustomOperators::Nop_1;
const static OperatorIdentifier PrintTensor  = CustomOperators::PrintTensor_1;
const static OperatorIdentifier ReduceMedian = CustomOperators::ReduceMedian;
const static OperatorIdentifier PackedDataBlock =
    CustomOperators::PackedDataBlock;
const static OperatorIdentifier ReplicatedAllReduce =
    CustomOperators::ReplicatedAllReduce;
const static OperatorIdentifier Remainder = CustomOperators::Remainder;
const static OperatorIdentifier Reshape   = CustomOperators::Reshape_1;
const static OperatorIdentifier Reverse   = CustomOperators::Reverse;
const static OperatorIdentifier Round     = CustomOperators::Round_1;
const static OperatorIdentifier Scale     = CustomOperators::Scale_1;
const static OperatorIdentifier ScaledAdd = CustomOperators::ScaledAdd;
const static OperatorIdentifier BinaryConstScalar =
    CustomOperators::BinaryConstScalar;
const static OperatorIdentifier ShapedDropout =
    CustomOperators::ShapedDropout_1;
const static OperatorIdentifier Subsample     = CustomOperators::Subsample_1;
const static OperatorIdentifier Abort         = CustomOperators::Abort;
const static OperatorIdentifier ScatterReduce = CustomOperators::ScatterReduce;
} // namespace OpSet1
} // namespace AiGraphcore

namespace CustomGradOperators {
const static AiGraphcoreOpIdV1
    DynamicUpdateToUpdateGrad("DynamicUpdateToUpdateGrad");
const static AiGraphcoreOpIdV1
    DynamicUpdateUpdaterGrad("DynamicUpdateUpdaterGrad");
const static AiGraphcoreOpIdV1
    NlllWithSoftmaxGradDirect("NlllWithSoftmaxGradDirect");
const static AiGraphcoreOpIdV1 AddBiasBiasGrad("AddBiasBiasGrad");
const static AiGraphcoreOpIdV1 AddBiasDataGrad("AddBiasDataGrad");
const static AiGraphcoreOpIdV1 CallGrad("CallGrad");
const static AiGraphcoreOpIdV1 ConstantLike("ConstantLike");
const static AiGraphcoreOpIdV1 DynamicSlicePadGrad("DynamicSlicePadGrad");
const static AiGraphcoreOpIdV1 DynamicZeroGrad("DynamicZeroGrad");
const static AiGraphcoreOpIdV1 IfConditionGrad("IfConditionGrad");
const static AiGraphcoreOpIdV1 IfGrad("IfGrad");
const static AiGraphcoreOpIdV1 L1Grad("L1Grad");
const static AiGraphcoreOpIdV1 NllGrad("NllGrad");
const static AiGraphcoreOpIdV1 CtcGrad("CtcGrad");
const static AiGraphcoreOpIdV1 ScatterReduceGradOp("ScatterReduceGradOp");
const static AiGraphcoreOpIdV1 SoftmaxGradDirect("SoftmaxGradDirect");
const static AiGraphcoreOpIdV1 SubsampleGrad("SubsampleGrad");
const static AiGraphcoreOpIdV1 UnaryZeroGradOp("UnaryZeroGrad");
} // namespace CustomGradOperators
} // namespace Onnx

} // namespace popart

#endif
