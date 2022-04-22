// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/tile.hpp>
#include <popart/popx/op/tilex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

void TileOpx::grow(snap::program::Sequence &prog) const {
  // not in-place, so cloning input
  auto outTensor =
      cloneNcopy(prog, getInTensor(TileOp::getInIndex())).getPoplarTensor();

  auto repeats = getOp<TileOp>().getRepeats();
  for (unsigned i = 0; i < repeats.size(); i++) {
    outTensor = outTensor.broadcast(static_cast<unsigned>(repeats[i]), i);
  }

  setOutTensor(TileOp::getOutIndex(), snap::Tensor{outTensor, graph()});
}

TileOpx::TileOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TileOp>(op);
}

TileGradOpx::TileGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TileGradOp>(op, Onnx::GradOperators::TileGrad);
}

// For non-zero repeat R, for corresponding dimension D:
// Take R equal slices, and sum the size(D)/R tensors, thereby reducing by
// factor R over D.
// e.g. for 1D case:
// GradIn [2, 6, 4, 7]
// Repeats [2]
// GradOut = Sum([2, 4], [6, 7]) = [8, 11]
void TileGradOpx::grow(snap::program::Sequence &prog) const {
  auto inTensor           = getInTensor(TileGradOp::getInIndex());
  auto intermediateTensor = inTensor;
  snap::Tensor outTensor;

  auto repeats = getOp<TileGradOp>().getRepeats();
  for (unsigned i = 0; i < repeats.size(); i++) {
    if (repeats[i] == 0)
      continue;

    // Do slice/sum reduction
    size_t inDimSize  = inTensor.dim(i);
    size_t outDimSize = inDimSize / repeats[i];
    for (size_t start = 0; start < inTensor.dim(i); start += outDimSize) {
      auto t = intermediateTensor.slice({start, start + outDimSize}, i);
      if (start == 0) {
        outTensor = cloneNcopy(prog, t);
      } else {
        snap::popops::mapInPlace(graph(),
                                 popops::expr::BinaryOpType::ADD,
                                 outTensor,
                                 t,
                                 prog,
                                 debugContext(std::string("reduceAdd") +
                                              sNameDelimiter +
                                              std::to_string(start)));
      }
    }
    intermediateTensor = outTensor;
  }

  setOutTensor(TileOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<TileOpx> tileOpxCreator({Onnx::Operators::Tile_1,
                                    Onnx::Operators::Tile_6});
OpxCreator<TileGradOpx> tileGradOpxCreator(Onnx::GradOperators::TileGrad);
} // namespace

} // namespace popx
} // namespace popart
