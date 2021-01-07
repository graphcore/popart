// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popart/error.hpp>
#include <popart/op/cast.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/castx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

CastOpx::CastOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CastOp>(op);
}

void CastOpx::grow(poplar::program::Sequence &prog) const {
  auto out = popops::cast(graph(),
                          getInTensor(CastOp::getInIndex()),
                          popType(op_p->outInfo(CastOp::getOutIndex())),
                          prog,
                          debugContext());

  if (hasInViewChangers(CastOp::getInIndex())) {
    setOutViewChangers(CastOp::getOutIndex(),
                       getInViewChangers(CastOp::getInIndex()));
  }

  setOutTensor(CastOp::getOutIndex(), out);
}

CastGradOpx::CastGradOpx(Op *op, Devicex *devicex) : CastOpx(op, devicex) {
  verifyOp<CastGradOp>(op, Onnx::GradOperators::CastGrad);
}

namespace {
OpxCreator<CastOpx> castOpxCreator({Onnx::Operators::Cast_1,
                                    Onnx::Operators::Cast_6,
                                    Onnx::Operators::Cast_9});
OpxCreator<CastGradOpx> castGradOpxCreator(Onnx::GradOperators::CastGrad);
} // namespace

} // namespace popx
} // namespace popart
