#include <poponnx/error.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/popx/op/gatherx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/util.hpp>

#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

// poplin::linspace only supports float or half :(
static poplar::Tensor
linspace(poplar::Graph &graph, int left, int right, int increment = 1) {
  std::size_t count = right - left;

  std::vector<int> values(count);
  std::iota(values.begin(), values.end(), 0);
  std::transform(values.begin(),
                 values.end(),
                 values.begin(),
                 [left, increment](int v) { return left + v * increment; });

  auto result =
      graph.addConstant(poplar::INT, {count}, poplar::ArrayRef<int>(values));

  graph.setTileMapping(result, 0);

  return result;
}

// Make b's rank match a.
//
// Assumes b.rank() <= a.rank() - dim
static poplar::Tensor
matchRank(poplar::Tensor a, poplar::Tensor b, unsigned dim) {
  std::vector<std::size_t> shape(a.rank(), 1);
  const auto b_shape = b.shape();

  std::copy(b_shape.begin(), b_shape.end(), shape.begin() + dim);

  return b.reshape(shape);
}

// Make b's shape match a.
//
// Assumes b is broadcastable into a
static poplar::Tensor broadcastShape(std::vector<std::size_t> a_shape,
                                     poplar::Tensor b) {
  for (int k = 0; k < a_shape.size(); ++k) {
    if (b.dim(k) == 1 && a_shape[k] != b.dim(k)) {
      b = b.broadcast(static_cast<unsigned>(a_shape[k]), k);
    }
  }

  return b;
}

// Make b's shape match a.
//
// Assumes b is broadcastable into a
static poplar::Tensor broadcastShape(poplar::Tensor a, poplar::Tensor b) {
  return broadcastShape(a.shape(), b);
}

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherOp>(op, Onnx::Operators::Gather_1);

  axis = dynamic_cast<GatherOp *>(op)->getAxis();
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(GatherOp::indicesInIndex());
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherOp::outIndex()));

  auto indices = get(inId(GatherOp::indicesInIndex()));
  auto data    = get(inId(GatherOp::dataInIndex()));

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().addVariable(data.elementType(), outputShape);

    insert(outId(GatherOp::outIndex()), result);
  } else {
    // Flatten the scalar indices
    indices                     = indices.flatten();
    const auto index_vector_dim = indices.rank();

    // Each slice is the shape of the data tensor
    // with the axis dimension set to 1
    std::vector<std::size_t> sliceSizes = data.shape();
    sliceSizes[axis]                    = 1;

    // All of the dimensions in the output, except the axis dimension are
    // offsets into the data tensor
    std::vector<std::size_t> offsetDims(data.rank());
    std::iota(offsetDims.begin(), offsetDims.begin() + axis, 0);
    std::iota(offsetDims.begin(), offsetDims.end(), axis + 1);
    std::vector<unsigned> startIndexMap = {static_cast<unsigned>(axis)};

    // Gather the slices
    auto result = popops::gather(graph(),
                                 data,
                                 indices,
                                 index_vector_dim,
                                 offsetDims,
                                 sliceSizes,
                                 {},
                                 startIndexMap,
                                 prog);

    // Shuffle and reshape into the expected ONNX shape
    std::vector<unsigned> result_shuffle(result.rank());
    std::iota(result_shuffle.begin(), result_shuffle.end(), 0);
    std::rotate(result_shuffle.begin(),
                result_shuffle.begin() + axis,
                result_shuffle.end());

    result = result.dimShuffle(result_shuffle);
    result = result.reshape(outputShape);

    // Reshape to the ONNX shape and insert the tensor
    insert(outId(GatherOp::outIndex()), result);
  }
}

GatherGradOpx::GatherGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherGradOp>(op, Onnx::GradOperators::GatherGrad);

  axis = dynamic_cast<GatherGradOp *>(op)->getAxis();
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update = get(inId(GatherGradOp::gradInIndex()));
  auto result = graph().addVariable(update.elementType(), outputShape);
  poputil::mapTensorLinearly(graph(), result);

  // Build the implicit index coordinates
  //
  // Create a grid of linspaced indices
  // Start by creating 1D linspaced constant tensors
  std::vector<poplar::Tensor> indices_mapped(result.rank());
  for (int i = 0; i < result.rank(); ++i) {
    indices_mapped[i] = linspace(graph(), 0, static_cast<int>(update.dim(i)));
  }

  // Insert the user provided indices
  indices_mapped[axis] = get(inId(GatherGradOp::indicesInIndex()));

  // Match the rank of the indices to the update tensor
  for (int i = 0; i < result.rank(); ++i) {
    indices_mapped[i] = matchRank(update, indices_mapped[i], i);
  }

  for (auto &index : indices_mapped) {
    // Match the shape of update
    index = broadcastShape(update, index);

    // Add a degenerate dimension for concatenation
    index = index.expand({index.rank()});
  }

  // Concat the grid of indices
  auto indices = poplar::concat(indices_mapped, update.rank());

  std::vector<unsigned> update_window_dims(update.rank());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(result.rank());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(update.rank());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  // Scatter the grad input into the result
  popops::scatter(graph(),
                  result,
                  indices,
                  update,
                  indices.rank() - 1,
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog);

  result = result.reshape(outputShape);

  insert(outId(GatherGradOp::gradOutIndex()), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator(Onnx::Operators::Gather_1);
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace poponnx
