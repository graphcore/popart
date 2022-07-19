// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_

#include <cstddef>
#include <map>

namespace snap {
class Graph;
class Tensor;
} // namespace snap

namespace popart {
namespace popx {

class LinearMapper {
private:
  class MapperImpl {
  public:
    void mapTensor(snap::Graph &graph, snap::Tensor &tensor);
    std::size_t next_mapping_start_index;
  };

public:
  void mapTensor(snap::Graph &graph, snap::Tensor &tensor);

private:
  std::map<snap::Graph *, MapperImpl> mappers;

  MapperImpl &getMapper(snap::Graph &graph);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_LINEARMAPPER_HPP_
