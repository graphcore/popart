// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VECTORANDSET_HPP
#define GUARD_NEURALNET_VECTORANDSET_HPP

#include <set>
#include <vector>

namespace popart {

// Stores elements in both a std::vector and a std::set
// The elements in the std::vector will be unique
template <class T> class VectorAndSet {
public:
  VectorAndSet();
  VectorAndSet(const std::vector<T> &vals);
  ~VectorAndSet();
  VectorAndSet &operator=(const VectorAndSet &rhs) = default;
  bool contains(T) const;
  const std::vector<T> &v() const;
  // insert string if not present, otherwise do nothing
  void insert(const T &);
  void reset(const std::vector<T> &vals);

private:
  std::vector<T> v_vals;
  std::set<T> m_vals;
};

template <class T> const std::vector<T> &VectorAndSet<T>::v() const {
  return v_vals;
}

template <class T> bool VectorAndSet<T>::contains(T name) const {
  return m_vals.count(name) == 1;
}

template <class T> VectorAndSet<T>::~VectorAndSet() = default;

template <class T> VectorAndSet<T>::VectorAndSet() {}

template <class T>
VectorAndSet<T>::VectorAndSet(const std::vector<T> &vals) : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

template <class T> void VectorAndSet<T>::reset(const std::vector<T> &vals) {

  // Replace the old with the new
  v_vals = vals;

  // Clear and initialise the m_vals set
  m_vals.clear();
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

template <class T> void VectorAndSet<T>::insert(const T &id) {
  if (m_vals.find(id) == m_vals.end()) {
    v_vals.push_back(id);
    m_vals.insert(id);
  }
}

} // namespace popart

#endif
