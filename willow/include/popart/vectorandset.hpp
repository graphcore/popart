// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VECTORANDSET_HPP
#define GUARD_NEURALNET_VECTORANDSET_HPP

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// Stores string elements in both a std::vector and a std::set
// The elements in the std::vector will be unique
class VectorAndSet {
public:
  VectorAndSet();
  VectorAndSet(const std::vector<std::string> &vals);
  ~VectorAndSet();
  VectorAndSet &operator=(const VectorAndSet &rhs) = default;
  bool contains(std::string) const;
  const std::vector<std::string> &v() const;
  // insert string if not present, otherwise do nothing
  void insert(const std::string &);
  void reset(const std::vector<std::string> &vals);

private:
  std::vector<std::string> v_vals;
  std::set<std::string> m_vals;
};

} // namespace popart

#endif
