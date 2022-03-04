// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/atan2.hpp>
#include <popart/popx/op/atan2x.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

Atan2Computex::Atan2Computex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

snap::Tensor Atan2Computex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &a,
                                     const snap::Tensor &b,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debugStr) const {
  return snap::Tensor{popops::atan2(graph.getPoplarGraph(),
                                    a.getPoplarTensor(),
                                    b.getPoplarTensor(),
                                    prog.getPoplarSequence(),
                                    {dnai, debugStr}),
                      graph};
}

void Atan2Computex::inplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tInOut,
                            const snap::Tensor &tIn,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &debugStr) const {
  popops::atan2InPlace(graph.getPoplarGraph(),
                       tInOut.getPoplarTensor(),
                       tIn.getPoplarTensor(),
                       prog.getPoplarSequence(),
                       {dnai, debugStr});
}

Atan2Opx::Atan2Opx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<Atan2Computex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<Atan2Op>(op, {Onnx::CustomOperators::Atan2_1});
}

Atan2LhsInplaceOpx::Atan2LhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<Atan2Computex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<Atan2LhsInplaceOp>(op);
}

namespace {
OpxCreator<Atan2Opx> atan2OpxCreator({Onnx::CustomOperators::Atan2_1});
OpxCreator<Atan2LhsInplaceOpx>
    atan2LhsInplaceOpxCreator(Onnx::CustomOperators::Atan2Inplace);

} // namespace

} // namespace popx
} // namespace popart
