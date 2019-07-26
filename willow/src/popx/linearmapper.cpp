#include <algorithm>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>

#include <popart/popx/linearmapper.hpp>

namespace popart {
namespace popx {

namespace {

template <typename T> void rotate_right(T &t, std::size_t spaces) {
  std::rotate(t.rbegin(), t.rbegin() + spaces, t.rend());
}

} // namespace

void LinearMapper::MapperImpl::mapTensor(poplar::Graph &graph,
                                         poplar::Tensor &tensor) {
  // poputil::calcLinearTileMapping always starts the mapping at tile0
  auto mapping = poputil::calcLinearTileMapping(graph, tensor);
  // the number of tiles the mapping is across
  auto mapping_tile_count = mapping.size();

  // shift the mapping to prevent always starting at tile 0
  auto tile_count = graph.getTarget().getNumTiles();
  mapping.resize(tile_count);
  rotate_right(mapping, next_mapping_start_index);

  next_mapping_start_index += mapping_tile_count;
  next_mapping_start_index = next_mapping_start_index % tile_count;

  graph.setTileMapping(tensor, mapping);
}

void LinearMapper::mapTensor(poplar::Graph &graph, poplar::Tensor &tensor) {
  auto &mapper = getMapper(graph);
  mapper.mapTensor(graph, tensor);
}

LinearMapper::MapperImpl &LinearMapper::getMapper(poplar::Graph &graph) {
  if (mappers.find(&graph) == mappers.end()) {
    mappers.insert({&graph, {}});
  }

  return mappers.at(&graph);
}

} // namespace popx
} // namespace popart
