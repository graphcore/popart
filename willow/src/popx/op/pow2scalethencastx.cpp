// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/ir.hpp"
#include <poplar/MetadataCreation.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/pow2scalethencastx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/pow2scalethencast.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
namespace popx {
class Devicex;

//
// *** Pow2ScaleThenCastOpx *** //
//

Pow2ScaleThenCastOpx::Pow2ScaleThenCastOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<Pow2ScaleThenCastOp>(op);
}

void Pow2ScaleThenCastOpx::grow(poplar::program::Sequence &prog) const {
  Pow2ScaleThenCastOp op = getOp<Pow2ScaleThenCastOp>();
  auto popartScaleTensor =
      getInTensor(Pow2ScaleThenCastOp::getlog2ScaleInIndex());

  if (op.getIr().getSessionOptions().throwIfLog2ScaleTensorNotInRange) {
    auto assertProg =
        createAssertLog2ScaleInRangeProg(graph(), popartScaleTensor, -32, 32);

    prog.add(assertProg);
  }
  // We need to negate the scale tensor here, to ensure we are always
  // multiplying by the scale factor. This "undoes" the negation done in
  // poplibs.
  popartScaleTensor = popops::map(graph(),
                                  pe::Mul(pe::_1, pe::Const(-1)),
                                  {popartScaleTensor},
                                  prog,
                                  debugContext());

  auto metadataTensor = poplar::createVariableMetadataTensor(
      graph(), getDestinationFormat(), popartScaleTensor, prog, debugContext());

  auto out = popops::cast(graph(),
                          getInTensor(Pow2ScaleThenCastOp::getInIndex()),
                          poplar::QUARTER,
                          metadataTensor,
                          prog,
                          debugContext());

  if (hasInViewChangers(Pow2ScaleThenCastOp::getInIndex())) {
    setOutViewChangers(Pow2ScaleThenCastOp::getOutIndex(),
                       getInViewChangers(Pow2ScaleThenCastOp::getInIndex()));
  }

  // Note: we reinterpret here as unsigned char, to avoid the requirement to
  // keep metadata with every quarter type tensor. Upon every float8 operation,
  // the tensor is reinterpreted as float8 with provided scale bias and format.
  setOutTensor(Pow2ScaleThenCastOp::getOutIndex(),
               out.reinterpret(poplar::UNSIGNED_CHAR));
}

poplar::QuarterMetadata::Format
Pow2ScaleThenCastOpx::getDestinationFormat() const {
  auto sourceFormat =
      op_p->outInfo(Pow2ScaleThenCastOp::getOutIndex()).dataType();

  if (sourceFormat == DataType::FLOAT8_143) {
    return poplar::QuarterMetadata::Format::F143;
  } else if (sourceFormat == DataType::FLOAT8_152) {
    return poplar::QuarterMetadata::Format::F152;
  } else {
    throw error("Format {} not supported for op {}.",
                op_p->outInfo(Pow2ScaleThenCastOp::getOutIndex()).data_type(),
                op_p->debugName());
  }
}

namespace {
OpxCreator<Pow2ScaleThenCastOpx>
    pow2ScaleThenCastOpxCreator({Onnx::CustomOperators::Pow2ScaleThenCast});
}

} // namespace popx
} // namespace popart
