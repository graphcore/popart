#include <poponnx/error.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/popx/op/gatherx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/util.hpp>

#include <popops/Gather.hpp>
#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherOp>(op, Onnx::Operators::Gather);

  axis = dynamic_cast<GatherOp *>(op)->getAxis();
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(GatherOp::indicesInIndex());
  const auto outputShape  = outShape(GatherOp::outIndex());
  const auto outputRank   = outputShape.size();

  // The size of each gathered slice
  auto sliceShape  = inShape(GatherOp::dataInIndex());
  sliceShape[axis] = 1;

  // The degenerate dimensions to squeeze out of the index tensor
  std::vector<std::size_t> indicesSqueezeDims;
  for (int i = 0; i < indicesShape.size(); ++i) {
    if (indicesShape[i] == 1) {
      indicesSqueezeDims.push_back(i);
    }
  }

  const auto indices =
      get(inId(GatherOp::indicesInIndex())).squeeze(indicesSqueezeDims);

  // If there are indices, return an empty tensor of the appropriate shape
  if (indices.numElements() == 0) {
    auto data = get(inId(GatherOp::dataInIndex()));

    auto result = graph().addVariable(data.elementType(),
                                      vXtoY<int64_t, std::size_t>(outputShape));
    poputil::mapTensorLinearly(graph(), result);

    insert(outId(GatherOp::outIndex()), result);
  } else {
    // The offset dimensions from the output tensor into the input tensor
    std::vector<std::size_t> offsetDims(outputRank - indices.rank() -
                                        indicesSqueezeDims.size());

    auto begin = offsetDims.begin();
    auto mid   = offsetDims.begin() + axis;
    auto end   = offsetDims.end();

    std::iota(begin, mid, 0);
    std::iota(mid, end, axis + indices.rank());

    // Gather the slices
    auto result = popops::gather(graph(),
                                 get(inId(GatherOp::dataInIndex())),
                                 indices,
                                 indices.rank(),
                                 offsetDims,
                                 vXtoY<int64_t, std::size_t>(sliceShape),
                                 {static_cast<std::size_t>(axis)},
                                 {static_cast<unsigned>(axis)},
                                 prog);

    // Reshape to the ONNX shape and insert the tensor
    insert(outId(GatherOp::outIndex()),
           result.reshape(vXtoY<int64_t, std::size_t>(outputShape)));
  }
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator(Onnx::Operators::Gather);
} // namespace

} // namespace popx
} // namespace poponnx
