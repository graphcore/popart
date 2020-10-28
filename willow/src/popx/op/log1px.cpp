// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

poplar::Tensor Log1pComputex::outplace(poplar::program::Sequence &p,
                                       poplar::Graph &g,
                                       const poplar::Tensor &t,
                                       const std::string &dbs) const {

  return popops::map(
      g, popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS, t, p, dbs);
}

void Log1pComputex::inplace(poplar::program::Sequence &p,
                            poplar::Graph &g,
                            const poplar::Tensor &t,
                            const std::string &dbs) const {

  popops::mapInPlace(
      g, popops::expr::UnaryOpType::LOGARITHM_ONE_PLUS, t, p, dbs);
}

namespace {
OpxCreator<Log1pOpx> log1pOpxCreator(Onnx::CustomOperators::Log1p_1);
OpxCreator<Log1pInplaceOpx>
    log1pInplaceOpxCreator(Onnx::CustomOperators::Log1pInplace);
} // namespace

} // namespace popx
} // namespace popart
