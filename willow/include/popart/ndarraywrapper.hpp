// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NDARRAYWRAPPER_HPP
#define GUARD_NEURALNET_NDARRAYWRAPPER_HPP
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>
#include <popart/iarray.hpp>
#include <popart/ndindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"

namespace popart {

template <typename T> class NDArrayWrapper : public IArray {
public:
  NDArrayWrapper(T *d, const TensorInfo &i);
  NDArrayWrapper(Tensor &tensor);
  NDArrayWrapper(T *d, const std::vector<int64_t> &shape);
  NDArrayWrapper(const NDArrayWrapper &) = default;

  T &operator[](int64_t i);
  const T &operator[](int64_t i) const;
  T &operator[](const std::vector<int64_t> &indices);
  const T &operator[](const std::vector<int64_t> &indices) const;

  std::vector<int64_t> unflatten(int64_t rem) const;
  int64_t flatten(const std::vector<int64_t> &indices) const;

  void *data() final;
  DataType dataType() const final;
  std::size_t rank() const final;
  int64_t dim(size_t index) const final;
  std::size_t nelms() const final;
  const Shape shape() const final;

private:
  T *data_;
  const TensorInfo info;
  const NDIndices ndindices;
};

template <typename T>
NDArrayWrapper<T>::NDArrayWrapper(T *d, const TensorInfo &i)
    : data_(d), info(i), ndindices(i) {}

template <typename T>
NDArrayWrapper<T>::NDArrayWrapper(Tensor &tensor)
    : NDArrayWrapper(reinterpret_cast<T *>(tensor.tensorData()->data()),
                     tensor.info) {}

template <typename T>
NDArrayWrapper<T>::NDArrayWrapper(T *d, const std::vector<int64_t> &shape)
    : NDArrayWrapper(d, TensorInfo(getDataType<T>(), shape)) {}

template <typename T> T &NDArrayWrapper<T>::operator[](int64_t i) {
  return data_[i];
}

template <typename T> const T &NDArrayWrapper<T>::operator[](int64_t i) const {
  return data_[i];
}

template <typename T>
T &NDArrayWrapper<T>::operator[](const std::vector<int64_t> &indices) {
  return data_[ndindices.flatten(indices)];
}

template <typename T>
const T &
    NDArrayWrapper<T>::operator[](const std::vector<int64_t> &indices) const {
  return data_[ndindices.flatten(indices)];
}

template <typename T>
std::vector<int64_t> NDArrayWrapper<T>::unflatten(int64_t rem) const {
  return ndindices.unflatten(rem);
}

template <typename T>
int64_t NDArrayWrapper<T>::flatten(const std::vector<int64_t> &indices) const {
  return ndindices.flatten(indices);
}

template <typename T> void *NDArrayWrapper<T>::data() {
  return reinterpret_cast<void *>(data_);
}

template <typename T> DataType NDArrayWrapper<T>::dataType() const {
  return info.dataType();
}

template <typename T> std::size_t NDArrayWrapper<T>::rank() const {
  return info.rank();
}

template <typename T> int64_t NDArrayWrapper<T>::dim(size_t index) const {
  return info.dim(static_cast<int>(index));
}

template <typename T> std::size_t NDArrayWrapper<T>::nelms() const {
  return info.nelms();
}

template <typename T> const Shape NDArrayWrapper<T>::shape() const {
  return info.shape();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const NDArrayWrapper<T> &array) {

  os << "{";
  for (int n = 0; n < array.nelms(); ++n) {
    if (n != 0) {
      os << ", ";
    }
    os << array[n];
  }
  os << "}";
  return os;
}

} // namespace popart

#endif
