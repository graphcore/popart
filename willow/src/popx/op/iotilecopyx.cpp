// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/iotilecopyx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

IoTileCopyOpx::IoTileCopyOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IoTileCopyOp>(op, Onnx::CustomOperators::IoTileCopy);
}

void IoTileCopyOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor outTensor = getOutTensor(IoTileCopyOp::getOutIndex());
  poplar::Tensor inView    = getInView(IoTileCopyOp::getInIndex());
  poplar::Tensor outView   = getOutView(IoTileCopyOp::getOutIndex());

  // Write undef the whole output tensor, which can be larger
  // than the getOutView tensor.
  poplar::program::WriteUndef writeUndef(outTensor);

  // Copy from view to view
  poplar::program::Copy outCopy(inView, outView);

  prog.add(writeUndef);
  prog.add(outCopy);
}

InputCreatorType IoTileCopyOpx::getInputCreatorType(InIndex index) const {
  // Currently, unwinding is only supported if the copy direction is
  // IOTile -> ComputeTile
  return index == IoTileCopyOp::getInIndex() &&
                 op_p->settings.tileSet == TileSet::Compute
             ? InputCreatorType::CanUnwind
             : Opx::getInputCreatorType(index);
}

poplar::Tensor IoTileCopyOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                 InIndex,
                                                 OutIndex) const {
  IoTileCopyOp &op = getOp<IoTileCopyOp>();
  auto info        = op.inInfo(IoTileCopyOp::getInIndex());

  // Source of unwinding (tensor originates from src graph)
  auto &srcGraph =
      dv_p->getVirtualGraph(getVirtualGraphId(), op_p->settings.tileSet);

  // Destination of unwinding (tensor unwound into dst graph)
  auto &dstGraph = dv_p->getVirtualGraph(
      getVirtualGraphId(),
      op_p->settings.tileSet == TileSet::Compute ? TileSet::IO
                                                 : TileSet::Compute);

  auto dstTensor = dstGraph.clone(tensor, "");

  auto numSrcTiles = srcGraph.getTarget().getNumTiles();
  auto numDstTiles = dstGraph.getTarget().getNumTiles();

  auto tilesPerTile = (numSrcTiles - 1) / numDstTiles + 1;

  auto srcTensorFlat = tensor.flatten();
  auto dstTensorFlat = dstTensor.flatten();

  // Reorder both tensors on the main graph
  dv_p->graph().reorderToSimplify(&srcTensorFlat, {&dstTensorFlat});

  auto srcMapping = srcGraph.getTileMapping(srcTensorFlat);
  poplar::Graph::TileToTensorMapping dstMapping(numDstTiles);

  for (size_t i = 0; i < srcMapping.size(); ++i) {
    auto j = i / tilesPerTile;
    dstMapping[j].insert(
        dstMapping[j].end(), srcMapping.at(i).begin(), srcMapping.at(i).end());
  }

  dstGraph.setTileMapping(dstTensorFlat, dstMapping);

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
