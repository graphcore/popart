// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_

#include <cstddef>
#include <map>

namespace poplar {
class Graph;
class Tensor;
} // namespace poplar

namespace popart {
namespace popx {

class LinearMapper {
private:
  class MapperImpl {
  public:
    void mapTensor(poplar::Graph &graph, poplar::Tensor &tensor);
    std::size_t next_mapping_start_index;
  };

public:
  void mapTensor(poplar::Graph &graph, poplar::Tensor &tensor);

private:
  std::map<poplar::Graph *, MapperImpl> mappers;

  MapperImpl &getMapper(poplar::Graph &graph);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_
