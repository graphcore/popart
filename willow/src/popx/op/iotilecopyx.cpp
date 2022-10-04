// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/iotilecopyx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {

namespace popx {

IoTileCopyOpx::IoTileCopyOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<IoTileCopyOp>(op, Onnx::CustomOperators::IoTileCopy);
}

void IoTileCopyOpx::grow(snap::program::Sequence &prog) const {
  snap::Tensor outTensor = getOutTensor(IoTileCopyOp::getOutIndex());
  snap::Tensor inView    = getInView(IoTileCopyOp::getInIndex());
  snap::Tensor outView   = getOutView(IoTileCopyOp::getOutIndex());

  // Write undef the whole output tensor, which can be larger
  // than the getOutView tensor.
  snap::program::WriteUndef writeUndef(outTensor, debugContext());

  // Copy from view to view
  snap::program::Copy outCopy(inView, outView, false, debugContext());

  prog.getPoplarSequence().add(writeUndef);
  prog.getPoplarSequence().add(outCopy);
}

InputCreatorType IoTileCopyOpx::getInputCreatorType(InIndex index) const {
  // Currently, unwinding is only supported if the copy direction is
  // IOTile -> ComputeTile
  return index == IoTileCopyOp::getInIndex() &&
                 op_p->settings.tileSet == TileSet::Compute
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor IoTileCopyOpx::unwindTensorLayout(snap::Tensor tensor,
                                               InIndex,
                                               OutIndex) const {
  IoTileCopyOp &op = getOp<IoTileCopyOp>();
  auto info        = op.inInfo(IoTileCopyOp::getInIndex());

  // Source of unwinding (tensor originates from src graph)
  auto &srcGraph = dv_p->lowering().getVirtualGraph(getVirtualGraphId(),
                                                    op_p->settings.tileSet);

  // Destination of unwinding (tensor unwound into dst graph)
  auto &dstGraph = dv_p->lowering().getVirtualGraph(
      getVirtualGraphId(),
      op_p->settings.tileSet == TileSet::Compute ? TileSet::IO
                                                 : TileSet::Compute);

  auto dstTensor = dstGraph.clone(tensor, "");

  auto numSrcTiles = srcGraph.getTarget().getNumTiles();
  auto numDstTiles = dstGraph.getTarget().getNumTiles();

  auto tilesPerTile = (numSrcTiles - 1) / numDstTiles + 1;

  auto srcTensorFlat = tensor.flatten().getPoplarTensor();
  auto dstTensorFlat = dstTensor.flatten();

  // Reorder both tensors on the main graph
  dv_p->lowering().graph().getPoplarGraph().reorderToSimplify(
      &srcTensorFlat, {&dstTensorFlat.getPoplarTensor()});

  auto srcMapping = srcGraph.getPoplarGraph().getTileMapping(srcTensorFlat);
  poplar::Graph::TileToTensorMapping dstMapping(numDstTiles);

  for (size_t i = 0; i < srcMapping.size(); ++i) {
    auto j = i / tilesPerTile;
    dstMapping[j].insert(
        dstMapping[j].end(), srcMapping.at(i).begin(), srcMapping.at(i).end());
  }

  dstGraph.getPoplarGraph().setTileMapping(dstTensorFlat.getPoplarTensor(),
                                           dstMapping);

  return dstTensor;
}

view::RegMap IoTileCopyOpx::unwindRegion(InIndex, OutIndex) const {
  auto info = inInfo(IoTileCopyOp::getInIndex());
  return [info](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<IoTileCopyOpx>
    IoTileCopyOpxCreator(Onnx::CustomOperators::IoTileCopy);
} // namespace

} // namespace popx
} // namespace popart
