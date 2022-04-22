// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VIEWCHANGERS_HPP
#define GUARD_NEURALNET_VIEWCHANGERS_HPP

#include <memory>
#include <snap/Tensor.hpp>
#include <typeinfo>
#include <vector>

namespace popart {
namespace popx {

// The identity view changer base class
class ViewChanger {
public:
  virtual ~ViewChanger() {}
  virtual snap::Tensor apply(snap::Tensor tensor) const { return tensor; }
  virtual bool containsAllDataRegions() const { return true; }
  virtual bool operator==(const ViewChanger &rhs) const {
    return typeid(&rhs) == typeid(ViewChanger);
  }
  virtual bool operator!=(const ViewChanger &rhs) const {
    return !(*this == rhs);
  }
};

// Chain of view changers
class ViewChangers {
public:
  ViewChangers();
  ViewChangers(std::vector<std::shared_ptr<ViewChanger>> viewChangers_);
  snap::Tensor apply(snap::Tensor tensor) const;
  bool empty() const { return viewChangers.empty(); }

  bool operator==(const ViewChangers &rhs) const;
  bool operator!=(const ViewChangers &rhs) const;

private:
  std::vector<std::shared_ptr<ViewChanger>> viewChangers;
};

} // namespace popx
} // namespace popart

#endif
