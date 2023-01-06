// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/ir.hpp"
#include <set>
#include <poplar/Graph.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/castthenpow2scalex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/castthenpow2scale.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace pe = popops::expr;

namespace popart {
namespace popx {

//
// *** CastThenPow2ScaleOpx *** //
//

CastThenPow2ScaleOpx::CastThenPow2ScaleOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<CastThenPow2ScaleOp>(op);
}

void CastThenPow2ScaleOpx::grow(poplar::program::Sequence &prog) const {
  CastThenPow2ScaleOp op = getOp<CastThenPow2ScaleOp>();
  auto popartScaleTensor =
      getInTensor(CastThenPow2ScaleOp::getlog2ScaleInIndex());

  if (op.getIr().getSessionOptions().throwIfLog2ScaleTensorNotInRange) {
    auto assertProg =
        createAssertLog2ScaleInRangeProg(graph(), popartScaleTensor, -32, 32);

    prog.add(assertProg);
  }

  // Note this input tensor is expected to be in poplar::UINT8 format. At the
  // popart level it is FLOAT8_152 or FLOAT8_143 data type.

  auto in             = getInTensor(CastThenPow2ScaleOp::getInIndex());
  auto metadataTensor = poplar::createVariableMetadataTensor(
      graph(), getSourceFormat(), popartScaleTensor, prog, debugContext());
  // We can't reinterpret to neither QUARTER_METADATA nor QUARTER type.
  // Instead, clone them and copy raw unsigned char data over.
  // This copy will be elided by poplar. We don't need to do this for the
  // metadata as we create it on the fly.
  auto q_data = graph().clone(poplar::QUARTER, metadataTensor.reshape({1}), in);

  prog.add(poplar::program::Copy(
      in, q_data.reinterpret(poplar::UNSIGNED_CHAR), false, debugContext()));

  // Finally cast to our output type.
  auto popartType = static_cast<CastThenPow2ScaleOp *>(op_p)->toDataType();

  auto out =
      popops::cast(graph(), q_data, popType(popartType), prog, debugContext());

  if (hasInViewChangers(CastThenPow2ScaleOp::getInIndex())) {
    setOutViewChangers(CastThenPow2ScaleOp::getOutIndex(),
                       getInViewChangers(CastThenPow2ScaleOp::getInIndex()));
  }

  setOutTensor(CastThenPow2ScaleOp::getOutIndex(), out);
}

poplar::QuarterMetadata::Format CastThenPow2ScaleOpx::getSourceFormat() const {
  auto sourceFormat =
      op_p->inInfo(CastThenPow2ScaleOp::getInIndex()).dataType();

  if (sourceFormat == DataType::FLOAT8_143) {
    return poplar::QuarterMetadata::Format::F143;
  } else if (sourceFormat == DataType::FLOAT8_152) {
    return poplar::QuarterMetadata::Format::F152;
  } else {
    throw error("Format {} not supported for op {}.",
                op_p->inInfo(CastThenPow2ScaleOp::getInIndex()).data_type(),
                op_p->debugName());
  }
}

std::set<TensorId>
CastThenPow2ScaleOpx::mustExistBeforeCreate(InIndex index) const {
  if (index == CastThenPow2ScaleOp::getInIndex()) {
    return {inId(CastThenPow2ScaleOp::getlog2ScaleInIndex())};
  }
  return {};
}

namespace {
OpxCreator<CastThenPow2ScaleOpx>
    castThenPow2ScaleOpxCreator({Onnx::CustomOperators::CastThenPow2Scale});
} // namespace

} // namespace popx
} // namespace popart
