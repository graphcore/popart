// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/log1p.hpp>
#include <popart/popx/op/log1px.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

Log1pInplaceOpx::Log1pInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, Log1pComputex::get()) {
  verifyOp<Log1pInplaceOp>(op, Onnx::CustomOperators::Log1pInplace);
}

Log1pOpx::Log1pOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, Log1pComputex::get()) {
  verifyOp<Log1pOp>(op, Onnx::CustomOperators::Log1p_1);
}

snap::Tensor Log1pComputex::outplace(snap::program::Sequence &p,
                                     snap::Graph &g,
                                     const snap::Tensor &t,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &dbs) const {

  return snap::Tensor{popops::map(g.getPoplarGraph(),
                                  popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS,
                                  t.getPoplarTensor(),
                                  p.getPoplarSequence(),
                                  {dnai, dbs}),
                      g};
}

void Log1pComputex::inplace(snap::program::Sequence &p,
                            snap::Graph &g,
                            const snap::Tensor &t,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &dbs) const {

  snap::popops::mapInPlace(
      g, popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS, t, p, {dnai, dbs});
}

namespace {
OpxCreator<Log1pOpx> log1pOpxCreator(Onnx::CustomOperators::Log1p_1);
OpxCreator<Log1pInplaceOpx>
    log1pInplaceOpxCreator(Onnx::CustomOperators::Log1pInplace);
} // namespace

} // namespace popx
} // namespace popart
