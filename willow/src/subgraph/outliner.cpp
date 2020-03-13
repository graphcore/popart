// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/subgraph/outliner.hpp>

namespace fwtools {
namespace subgraph {

OutlinerAlgorithm getDefaultOutlinerAlgorithm() {
  return OutlinerAlgorithm::ALGO1;
}

} // namespace subgraph
} // namespace fwtools
