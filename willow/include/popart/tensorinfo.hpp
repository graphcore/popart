// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#include <algorithm>
#include <sstream>
#include <vector>
#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>

namespace popart {

// There is a 1-1 correspondence
// between popart::DataTypes
// and ONNX_NAMESPACE::TensorProto_DataTypes, aka
// decltype(ONNX_NAMESPACE::TensorProto().data_type()).

enum class DataType {
  // fixed point types
  UINT8 = 0,
  INT8,
  UINT16,
  INT16,
  INT32,
  INT64,
  UINT32,
  UINT64,
  BOOL,
  // floating point types
  FLOAT,
  FLOAT16,
  BFLOAT16,
  DOUBLE,
  COMPLEX64,
  COMPLEX128,
  // other types
  STRING,
  UNDEFINED,
};

template <typename T> DataType getDataType();
template <> DataType getDataType<int8_t>();
template <> DataType getDataType<int16_t>();
template <> DataType getDataType<int32_t>();
template <> DataType getDataType<int64_t>();
template <> DataType getDataType<uint8_t>();
template <> DataType getDataType<uint16_t>();
template <> DataType getDataType<uint32_t>();
template <> DataType getDataType<uint64_t>();
template <> DataType getDataType<bool>();
template <> DataType getDataType<Half>();
template <> DataType getDataType<float>();
template <> DataType getDataType<double>();
template <> DataType getDataType<std::string>();

class TensorInfo;

// Check if two tensors can be (numpy) broadcasted based on
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
bool npBroadcastable(const std::vector<int64_t> &s0,
                     const std::vector<int64_t> &s1);
bool npBroadcastable(const std::vector<int64_t> &s0,
                     const std::vector<int64_t> &s1,
                     size_t &overlap);
bool npBroadcastable(const TensorInfo &i0, const TensorInfo &i1);

// Calculate the numpy broadcast shape as described in
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
std::vector<int64_t> npOut(const std::vector<int64_t> &s0,
                           const std::vector<int64_t> &s1,
                           const std::string &debugName = "");

// Compute the reduction axis for a reduction op.
std::vector<int64_t> npReductionAxis(const std::vector<int64_t> &in,
                                     const std::vector<int64_t> &out);

template <typename T> std::vector<T> squeeze(const std::vector<T> &v) {
  std::vector<T> w;
  w.reserve(v.size());
  for (auto &x : v) {
    if (x != 1) {
      w.push_back(x);
    }
  }
  return w;
}

template <typename T>
std::vector<T> squeeze(const std::vector<T> &v, const std::vector<T> &axes) {
  std::vector<T> new_shape;
  new_shape.reserve(v.size());
  for (int i = 0; i < v.size(); i++) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      new_shape.push_back(v[i]);
    }
  }

  return new_shape;
}

template <typename T>
std::vector<T> unsqueeze(const std::vector<T> &v, const std::vector<T> &axes) {
  if (!std::is_sorted(axes.begin(), axes.end())) {
    throw error("`axes' input to `unsqueeze' function must be sorted");
  }

  std::vector<T> new_shape;
  new_shape.reserve(v.size() + axes.size());

  auto it = v.begin();

  for (int i = 0; i < v.size() + axes.size(); i++) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      new_shape.push_back(*it);
      it++;
    } else {
      new_shape.push_back(1);
    }
  }

  return new_shape;
}

// Check if two tensors can be (numpy) broadcasted based on
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
bool npBroadcastable(const TensorInfo &i0, const TensorInfo &i1);

// Calculate the numpy broadcast shape as described in
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
TensorInfo npOut(const TensorInfo &i0, const TensorInfo &i1);

// FLOAT, FLOAT16, INT8 etc.
class DataTypeInfo {
public:
  DataTypeInfo(DataType type__,
               int nbytes__,
               bool isFixedPoint__,
               std::string name__,
               std::string lcasename__);
  DataType type() const;
  // number of bytes of 1 element
  const int &nbytes() const;
  const std::string &name() const;
  const std::string &lcasename() const;
  bool isFixedPoint() const;

private:
  DataType type_;
  int nbytes_;
  bool isFixedPoint_;
  std::string name_;
  std::string lcasename_;
};

const std::map<DataType, DataTypeInfo> &getDataTypeInfoMap();
std::map<DataType, DataTypeInfo> initDataTypeInfoMap();

const std::map<std::string, DataType> &getStrToDataTypeMap();
std::map<std::string, DataType> initStrToDataTypeMap();

const std::string &getAllONNXTypesString();
std::string initAllONNXTypesString();

DataType dataTypeFromString(const std::string &s);

class TensorInfo {
public:
  TensorInfo(DataType, const Shape &);
  TensorInfo(std::string data_type, std::string shape);
  TensorInfo(std::string data_type, const Shape &);
  explicit TensorInfo(const ONNX_NAMESPACE::TensorProto &);
  explicit TensorInfo(const ONNX_NAMESPACE::TypeProto &);
  void set(const ONNX_NAMESPACE::TensorProto &);
  void set(const ONNX_NAMESPACE::TypeProto &);
  TensorInfo() = default;
  void set(DataType, const Shape &);
  const Shape &shape() const;
  // A helper functions for back-ends which
  // prefer the size as (unsigned) size_t.
  std::vector<size_t> shape_szt() const;
  Rank rank() const;
  int64_t nelms() const;
  // total bytes of tensor
  int64_t nbytes() const;
  int64_t dim(int i) const;
  DataType dataType() const;
  const std::string &data_type() const;
  const std::string &data_type_lcase() const;
  void append(std::ostream &) const;
  bool isSet() const;
  bool operator==(const TensorInfo &) const;
  bool operator!=(const TensorInfo &) const;
  Shape shapeFromString(const std::string &s) const;
  ONNX_NAMESPACE::TypeProto getOnnxTypeProto() const;
  const DataTypeInfo *getDataTypeInfo() const;

private:
  const DataTypeInfo *dataTypeInfo = nullptr;
  Shape shape_v;
};

std::ostream &operator<<(std::ostream &stream, const TensorInfo &ti);
std::ostream &operator<<(std::ostream &stream, const DataType &dt);

} // namespace popart

#endif
