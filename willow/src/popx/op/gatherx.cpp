#include <poponnx/error.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/gatherx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

namespace poponnx {
namespace popx {

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherOp>(op, Onnx::Operators::Gather_1);

  axis = dynamic_cast<GatherOp *>(op)->getAxis();
}

// Given a tensor with a dimension that has an odd size, pad it with zeros to
// make it even
static poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t, unsigned dim) {
  if (t.dim(dim) % 2) {
    auto shape = t.shape();
    shape[dim] = 1;

    auto zero_slice = graph.addConstant(t.elementType(), shape, 0, "/zero");
    graph.setTileMapping(zero_slice, 0);

    return poplar::concat(t, zero_slice, dim);
  } else {
    return t;
  }
}

// Split a tensors dimension in half
//
// Assumes the given dimension has even size
static poplar::Tensor splitOperand(poplar::Tensor t, unsigned dim) {
  if (t.dim(dim) % 2) {
    throw error("Cannot evenly split tensor on dimension {} with size {}",
                dim,
                t.dim(dim));
  }

  auto shape = t.shape();

  shape[dim] = t.dim(dim) / 2;
  shape.insert(shape.begin() + dim, 2);

  return t.reshape(shape);
}

// Given a tensor that has been split with `splitOperand`, update the indices to
// correct map into the new tensor
static poplar::Tensor splitIndices(poplar::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   poplar::Tensor t,
                                   unsigned dim,
                                   poplar::Tensor i,
                                   std::size_t idx) {
  if (i.rank() == 1) {
    return splitIndices(graph, prog, t, dim, i.expand({1}), idx);
  } else if (i.rank() == 2) {
    auto one = graph.addConstant(i.elementType(), {}, 1, "/one");
    graph.setTileMapping(one, 0);

    i        = i.transpose();
    auto i_d = i[idx];

    auto i_m = popops::bitwiseAnd(graph, i_d, one, prog); // i_m = i_d mod 2
    popops::shiftRightInPlace(graph, i_d, one, prog);     // i_d /= 2

    i_m = i_m.expand({0});

    auto prefix = i.slice(0, idx + 1, 0);
    auto suffix = i.slice(idx + 1, i.dim(0), 0);

    return poplar::concat({prefix, i_m, suffix}, 0).transpose();
  } else {
    throw error("Cannot split indices with rank greater than 2, the split "
                "dimension is ambiguous");
  }
}

// We will be slicing on the new dimension, so it's size will be 1
static std::vector<std::size_t>
splitSliceSizes(std::vector<std::size_t> sliceSizes, unsigned dim) {
  sliceSizes.insert(sliceSizes.begin() + dim + 1, 1);

  return sliceSizes;
}

// The new dimension will be slice on, collapsing it will keep the output shape
// unchanged
static std::vector<std::size_t>
splitCollapsedSliceDims(std::vector<std::size_t> collapsedSliceDims,
                        unsigned dim) {
  for (auto &collapsed_dim : collapsedSliceDims) {
    if (collapsed_dim > dim) {
      collapsed_dim += 1;
    }
  }

  collapsedSliceDims.push_back(dim + 1);
  return collapsedSliceDims;
}

// The starting indices now need to point at the changed dimensions.
// Also need to insert a new index for the split indices tensor.
static std::vector<unsigned>
splitStartIndexMap(std::vector<unsigned> startIndexMap,
                   unsigned dim,
                   std::size_t idx) {
  for (auto &start_index : startIndexMap) {
    if (start_index > dim) {
      start_index += 1;
    }
  }

  startIndexMap.insert(startIndexMap.begin() + idx + 1, dim + 1);

  return startIndexMap;
}

// Adapt a general gather into a poplar compatible gather.
//
// This means none of the startIndexMap dimensions can be larger than 2^16
//
// A brief overview of what I think the algorithm to solve this should be.
//
// Step 1. Pad the tensor in the axis dimension to have even length.
// Step 2. Split the axis dimension into two and recompute the indices
// Step 3. Repeat until all axis dimensions fit
static poplar::Tensor gatherWrapper(poplar::Graph &graph,
                                    poplar::Tensor operand,
                                    poplar::Tensor indices,
                                    std::size_t indexVectorDim,
                                    std::vector<std::size_t> offsetDims,
                                    std::vector<std::size_t> sliceSizes,
                                    std::vector<std::size_t> collapsedSliceDims,
                                    std::vector<unsigned> startIndexMap,
                                    poplar::program::Sequence &prog,
                                    const std::string &debugPrefix = "") {
  const auto too_big_dim = boost::range::find_if(
      startIndexMap, [&](unsigned dim) { return operand.dim(dim) > 0xFFFF; });

  if (too_big_dim == startIndexMap.end()) {
    return popops::gather(graph,
                          operand,
                          indices,
                          indexVectorDim,
                          offsetDims,
                          sliceSizes,
                          collapsedSliceDims,
                          startIndexMap,
                          prog,
                          debugPrefix);
  } else {
    const auto index = std::distance(startIndexMap.begin(), too_big_dim);

    // Step 1. Pad the tensor in the axis dimension to have even length
    operand = pad(graph, operand, *too_big_dim);

    // Step 2. Split the axis dimension into two and recompute the indices
    operand = splitOperand(operand, *too_big_dim);
    indices = splitIndices(graph, prog, operand, *too_big_dim, indices, index);
    sliceSizes = splitSliceSizes(sliceSizes, *too_big_dim);
    collapsedSliceDims =
        splitCollapsedSliceDims(collapsedSliceDims, *too_big_dim);
    startIndexMap = splitStartIndexMap(startIndexMap, *too_big_dim, index);

    // Step 3. Repeat until all axis dimensions fit
    return gatherWrapper(graph,
                         operand,
                         indices,
                         indexVectorDim,
                         offsetDims,
                         sliceSizes,
                         collapsedSliceDims,
                         startIndexMap,
                         prog,
                         debugPrefix);
  }
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(GatherOp::indicesInIndex());
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherOp::outIndex()));

  auto indices = getInTensor(GatherOp::indicesInIndex());
  auto data    = getInTensor(GatherOp::dataInIndex());

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().addVariable(data.elementType(), outputShape);

    setOutTensor(GatherOp::outIndex(), result);
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
    std::iota(offsetDims.begin() + axis, offsetDims.end(), axis + 1);
    std::vector<unsigned> startIndexMap = {static_cast<unsigned>(axis)};

    // Gather the slices
    auto result = gatherWrapper(graph(),
                                data,
                                cloneNcopy(prog, indices),
                                index_vector_dim,
                                offsetDims,
                                sliceSizes,
                                {},
                                startIndexMap,
                                prog);

    // Reshape into the expected ONNX shape
    result = result.reshape(outputShape);

    setOutTensor(GatherOp::outIndex(), result);
  }
}

static std::size_t findG(std::size_t d, std::size_t t) {
  for (auto g = (d + t - 1) / t; g <= d; ++g) {
    if (d % g == 0) {
      return g;
    }
  }

  throw error("Cannot find a value of G that is both a factor of D and "
              "satisfies D / G <= T");
}

static std::size_t quotCeil(std::size_t a, std::size_t b) {
  return (a + b - 1) / b;
}

static std::size_t findH(std::size_t d, std::size_t t) {
  return std::max<std::size_t>(1, t / d);
}

static poplar::Tensor createGatherInput(poplar::Graph &graph,
                                        int64_t axis,
                                        const poplar::Type &type,
                                        const std::vector<std::size_t> &shape,
                                        const std::string debugName) {
  const auto volume =
      boost::accumulate(shape, 1, std::multiplies<std::size_t>());

  // Number of tiles
  const auto t = graph.getTarget().getNumTiles();

  // Size of the "sequence" dimension
  const auto s = shape[axis];

  // Product of other n-1 dimensions
  const auto d = volume / s;

  // Grouping factor
  const auto g = findG(d, t);

  // Balance factor
  const auto h = (g == 1) ? findH(d, t) : 1;

  // We will allocate and map so that each tile gets s*g/h elements
  // The rounding in s/h will introduce some padding
  const std::vector<std::size_t> allocShape = {d / g, h, quotCeil(s, h), g};

  auto result = graph.addVariable(type, allocShape, debugName);

  result = result.reshape({h * d / g, quotCeil(s, h), g});
  for (std::size_t i = 0; i < result.dim(0); ++i) {
    graph.setTileMapping(result[i], static_cast<unsigned>(i));
  }
  result = result.reshape(allocShape);
  result = result.dimShuffle({0, 3, 1, 2});

  // Reshape back into the desired shape
  auto tmp_shape  = shape;
  tmp_shape[axis] = h * quotCeil(s, h);
  std::swap(tmp_shape[axis], tmp_shape.back());
  result = result.reshape(tmp_shape);

  std::vector<unsigned> permutation(result.rank());
  boost::iota(permutation, 0);
  std::swap(permutation[axis], permutation.back());
  result = result.dimShuffle(permutation);

  // Slice away any padding
  return result.slice(0, shape[axis], static_cast<unsigned>(axis));
}

poplar::Tensor GatherOpx::createInput(int index,
                                      const std::string &name) const {
  if (index != GatherOp::dataInIndex()) {
    throw error("GatherOpx::createInput Cannot create input {}", index);
  }

  auto info = inInfo(GatherOp::dataInIndex());

  const auto shape = info.shape_szt();

  return createGatherInput(graph(), axis, popType(info), shape, name);
}

InputCreatorType GatherOpx::getInputCreatorType(int index0) const {
  return index0 == GatherOp::dataInIndex() ? InputCreatorType::CANCREATE
                                           : Opx::getInputCreatorType(index0);
}

bool GatherOpx::createsEquiv(int, Opx *, int) const { return false; }

std::vector<TensorId> GatherOpx::mustExistBeforeCreate(int) const { return {}; }

GatherGradOpx::GatherGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherGradOp>(op, Onnx::GradOperators::GatherGrad);

  axis = dynamic_cast<GatherGradOp *>(op)->getAxis();
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update = getInTensor(GatherGradOp::gradInIndex());
  auto result = createGatherInput(
      graph(),
      axis,
      update.elementType(),
      outputShape,
      op_p->str() + "output" + std::to_string(GatherGradOp::gradOutIndex()));

  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  std::vector<unsigned> update_window_dims(update.rank() - indices.rank());
  auto begin = update_window_dims.begin();
  auto mid   = update_window_dims.begin() + axis;
  auto end   = update_window_dims.end();
  std::iota(begin, mid, 0);
  std::iota(mid, end, axis + indices.rank());

  std::vector<std::size_t> inserted_window_dims = {
      static_cast<std::size_t>(axis)};

  std::vector<unsigned> scatter_dims_to_op = {static_cast<unsigned>(axis)};

  // Add overlapping gradients
  popops::UpdateComputationFunc updateComp =
      [](poplar::Graph &g,
         poplar::Tensor &a,
         poplar::Tensor &b,
         poplar::program::Sequence &p) -> poplar::Tensor {
    popops::addInPlace(g, b, a, p);

    return b;
  };

  // Scatter the grad input into the result
  popops::scatter(graph(),
                  result,
                  indices,
                  update,
                  indices.rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  updateComp,
                  prog);

  result = result.reshape(outputShape);

  setOutTensor(GatherGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator(Onnx::Operators::Gather_1);
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace poponnx
