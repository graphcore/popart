// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/popx/viewchangers.hpp>

namespace popart {
namespace popx {

poplar::Tensor ViewChangers::apply(poplar::Tensor tensor) const {
  for (auto viewChanger : viewChangers) {
    tensor = viewChanger->apply(tensor);
  }
  return tensor;
}

ViewChangers::ViewChangers() {}

ViewChangers::ViewChangers(
    std::vector<std::shared_ptr<ViewChanger>> viewChangers_)
    : viewChangers(viewChangers_){};

} // namespace popx
} // namespace popart
