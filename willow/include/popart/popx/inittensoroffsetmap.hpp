// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_INITTENSOROFFSETMAP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_INITTENSOROFFSETMAP_HPP_

#include <cstddef>
#include <map>

namespace poplar {
class Graph;
class Tensor;
} // namespace poplar

namespace popart {
namespace popx {

class InitTensorOffsetMap {
public:
  std::size_t getOffset(poplar::Graph &graph);
  void setOffset(poplar::Graph &graph, const std::size_t offset);

private:
  // offset, the created tensor bytes
  std::map<poplar::Graph *, std::size_t> offsets;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_INITTENSOROFFSETMAP_HPP_
