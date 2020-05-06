// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VIEWCHANGERS_HPP
#define GUARD_NEURALNET_VIEWCHANGERS_HPP

#include <vector>
#include <poplar/Tensor.hpp>

namespace popart {
namespace popx {

// The identity view changer base class
class ViewChanger {
public:
  virtual ~ViewChanger() {}
  virtual poplar::Tensor apply(poplar::Tensor tensor) const;
  virtual bool containsAllDataRegions() const { return true; }
};

// Chain of view changers
class ViewChangers {
public:
  ViewChangers();
  ViewChangers(std::vector<std::shared_ptr<ViewChanger>> viewChangers_);
  poplar::Tensor apply(poplar::Tensor tensor) const;
  bool empty() const { return viewChangers.empty(); }

private:
  std::vector<std::shared_ptr<ViewChanger>> viewChangers;
};

} // namespace popx
} // namespace popart

#endif
