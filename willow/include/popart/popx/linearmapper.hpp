// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LINEARMAPPER_HPP
#define GUARD_NEURALNET_LINEARMAPPER_HPP

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

#endif
