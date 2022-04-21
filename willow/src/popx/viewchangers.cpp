// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/popx/viewchangers.hpp>

namespace popart {
namespace popx {

snap::Tensor ViewChangers::apply(snap::Tensor tensor) const {
  for (auto viewChanger : viewChangers) {
    tensor = viewChanger->apply(tensor);
  }
  return tensor;
}

bool ViewChangers::operator==(const ViewChangers &rhs) const {
  if (empty() != rhs.empty() ||
      viewChangers.size() != rhs.viewChangers.size()) {
    return false;
  }

  for (size_t i = 0; i < viewChangers.size(); ++i) {
    if (*viewChangers.at(i) != *rhs.viewChangers.at(i)) {
      return false;
    }
  }
  return true;
}

bool ViewChangers::operator!=(const ViewChangers &rhs) const {
  return !(*this == rhs);
}

ViewChangers::ViewChangers() {}

ViewChangers::ViewChangers(
    std::vector<std::shared_ptr<ViewChanger>> viewChangers_)
    : viewChangers(viewChangers_) {}

} // namespace popx
} // namespace popart
