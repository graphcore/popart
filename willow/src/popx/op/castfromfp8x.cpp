// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popart/error.hpp>
#include <popart/op/castfromfp8.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/castfromfp8x.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popfloat/experimental/CastToGfloat.hpp>
using namespace popfloat::experimental;

namespace popart {
namespace popx {

CastFromFp8Opx::CastFromFp8Opx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<CastFromFp8Op>(op);
}

void CastFromFp8Opx::grow(snap::program::Sequence &prog) const {
  CastFromFp8Op &op      = getOp<CastFromFp8Op>();
  bool enableDenorms     = true;
  bool enableInfsAndNans = false;
  int man                = op.getNBitMantissa();
  int exp                = op.getNBitExponent();
  int bias               = op.getExponentBias();
  DataType toDataType    = op.getDataType();
  SpecType specCalculationType;
  if (toDataType == DataType::FLOAT)
    specCalculationType = SpecType::FP32;
  else if (toDataType == DataType::FLOAT16)
    specCalculationType = SpecType::FP16;
  else
    throw error("CastFromFp8Opx does not support output data type : {}",
                toDataType);

  auto gfFormatCfg = GfloatCast::FormatConfig(
      man, exp, bias, enableDenorms, enableInfsAndNans, specCalculationType);
  // NOT used, only for popfloat::experimental::RoundType::SR
  int numberSRBits = 23;
  auto roundCfg = GfloatCast::RoundConfig(popfloat::experimental::RoundType::RN,
                                          numberSRBits,
                                          gfFormatCfg.getCalculationType());
  bool enableNanoo     = false;
  bool enableNanooMode = enableNanoo && enableInfsAndNans && (exp > 0);
  auto gfCast =
      GfloatCast(gfFormatCfg,
                 roundCfg,
                 enableNanooMode,
                 // castNativeToGfloat output storage type
                 SpecType::INT8,
                 // the native floating point Type used to store the
                 // gfloat format when casting a Gfloat to a native type
                 specCalculationType);

  gfCast.createCastOpParamsTensor(graph().getPoplarGraph(),
                                  prog.getPoplarSequence(),
                                  debugContext("CastFromFp8/param"));

  const poplar::Tensor &in =
      getInTensor(CastFromFp8Op::getInIndex()).getPoplarTensor();
  auto out = gfCast.castGfloatToNative(graph().getPoplarGraph(),
                                       in,
                                       prog.getPoplarSequence(),
                                       debugContext("CastFromFp8/cast"));
  setOutTensor(CastFromFp8Op::getOutIndex(), snap::Tensor{out, graph()});
}

namespace {
OpxCreator<CastFromFp8Opx>
    CastFromFp8OpxCreator({Onnx::CustomOperators::CastFromFp8});
} // namespace

} // namespace popx
} // namespace popart
