// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popart/error.hpp>
#include <popart/op/casttofp8.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/casttofp8x.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popfloat/experimental/CastToGfloat.hpp>
using namespace popfloat::experimental;

namespace popart {
namespace popx {

CastToFp8Opx::CastToFp8Opx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<CastToFp8Op>(op);
}

void CastToFp8Opx::grow(snap::program::Sequence &prog) const {
  CastToFp8Op &op        = getOp<CastToFp8Op>();
  bool enableDenorms     = true;
  bool enableInfsAndNans = false;
  int man                = op.getNBitMantissa();
  int exp                = op.getNBitExponent();
  int bias               = op.getExponentBias();

  const poplar::Tensor &in =
      getInTensor(CastToFp8Op::getInIndex()).getPoplarTensor();

  poplar::Type fromDataType = in.elementType();
  SpecType specCalculationType;
  if (fromDataType == poplar::FLOAT)
    specCalculationType = SpecType::FP32;
  else if (fromDataType == poplar::HALF)
    specCalculationType = SpecType::FP16;
  else
    throw error("CastToFp8Opx does not support input data type : {}",
                fromDataType.toString());

  auto gfFormatCfg = GfloatCast::FormatConfig(
      man, exp, bias, enableDenorms, enableInfsAndNans, specCalculationType);
  // NOT used, only  for popfloat::experimental::RoundType::SR
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
                                  debugContext("CastToFp8/param"));

  auto out = gfCast.castNativeToGfloat(graph().getPoplarGraph(),
                                       in,
                                       prog.getPoplarSequence(),
                                       debugContext("CastToFp8/cast"));

  setOutTensor(CastToFp8Op::getOutIndex(), snap::Tensor{out, graph()});
}

namespace {
OpxCreator<CastToFp8Opx>
    CastToFp8OpxCreator({Onnx::CustomOperators::CastToFp8});
} // namespace

} // namespace popx
} // namespace popart
