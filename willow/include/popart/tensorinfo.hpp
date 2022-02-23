// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>
#include <popart/basicoptionals.hpp>
#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>

namespace popart {

/// There is a one-to-one correspondence
/// between \c popart::DataTypes
/// and \c ONNX_NAMESPACE::TensorProto_DataTypes, or
/// \c decltype(ONNX_NAMESPACE::TensorProto().data_type()).
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

using OptionalDataType = BasicOptional<DataType, 0>;

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

/// Return the IPU-compatible data type of any data type
/// (Hardware) unsupported data types that have a compatible counterpart:
/// INT64 -> INT32
/// UINT64 -> UINT32
DataType getCompatibleDataType(DataType type);

/// Check if two tensors can be (numpy) broadcasted based on
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
bool npBroadcastable(const std::vector<int64_t> &s0,
                     const std::vector<int64_t> &s1);
bool npBroadcastable(const std::vector<int64_t> &s0,
                     const std::vector<int64_t> &s1,
                     size_t &overlap);

/// Calculate the numpy broadcast shape as described in
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
std::vector<int64_t> npOut(const std::vector<int64_t> &s0,
                           const std::vector<int64_t> &s1,
                           const std::string &debugName = "");

/// Returns the DataType of the output tensor of a numpy broadcast
/// operation.
///
/// Note: the inferred DataType matches that of Poplar, which is not
/// necessarily the same as that inferred by numpy.
DataType getOutputDataType(const TensorInfo &i0,
                           const TensorInfo &i1,
                           const std::string &debugName);

/// Calculate the numpy broadcast shape as described in
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
///
/// Note: The inferred DataType of the returned TensorInfo is that of Poplar,
/// and does not necessarily match that of numpy.
TensorInfo npOut(const TensorInfo &i0,
                 const TensorInfo &i1,
                 bool checkDataType,
                 const std::string &debugName = "");

/// Compute the reduction axis for a reduction op.
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

int64_t getONNXDataTypeAsInt(const DataType dtype);

class TensorInfo {
public:
  /// Create TensorInformation based on data type and shape.
  ///
  /// \param data_type    - The data type.
  /// \param shape        - The actual shape of the tensor.
  TensorInfo(DataType, const Shape &);
  /// Create TensorInformation based on data type, shape and meta shape
  ///
  /// \param data_type    - The data type.
  /// \param shape        - The actual shape of the tensor.
  /// \param meta_shape   - The meta shape of the tensor, which can for example
  ///                       be used to store the original tensor shape before
  ///                       replicated tensor sharding was applied.
  TensorInfo(DataType data_type, const Shape &shape, const Shape &meta_shape);
  TensorInfo(std::string data_type, std::string shape);
  TensorInfo(std::string data_type, const Shape &);
  explicit TensorInfo(const ONNX_NAMESPACE::TensorProto &);
  explicit TensorInfo(const ONNX_NAMESPACE::TypeProto &);
  void set(const ONNX_NAMESPACE::TensorProto &);
  void set(const ONNX_NAMESPACE::TypeProto &);
  TensorInfo() = default;
  void set(DataType);
  void set(DataType, const Shape &);
  void set(DataType, const Shape &, const Shape &);
  const Shape &shape() const;
  //
  const Shape &metaShape() const;
  // A helper functions for back-ends which
  // prefer the size as (unsigned) size_t.
  std::vector<size_t> shape_szt() const;
  // Defined in-header to encourage inlining (it's called a lot).
  Rank rank() const { return static_cast<int>(shape_v.size()); }
  // Defined in-header to encourage inlining (it's called a lot).
  int64_t nelms() const {
    return std::accumulate(shape_v.begin(),
                           shape_v.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }
  // total bytes of tensor
  int64_t nbytes() const;
  // Defined in-header to encourage inlining (it's called a lot).
  int64_t dim(int i) const {
    if (i >= shape_v.size()) {
      throw error(
          "Invalid input dimension {}, tensor of rank {}", i, shape_v.size());
    }
    return shape_v[i];
  }

  /**
   * Get the strides of the tensor, that is the number of bytes to step in each
   * dimension when traversing an array in memory. See
   * https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
   *
   * \returns std::vector<int> The strides vector.
   */
  std::vector<int> strides() {
    auto shape = this->shape();

    if (shape.size() == 0) {
      return {};
    } else if (shape.size() == 1 && shape.at(0) == 0) {
      return {this->getDataTypeInfo()->nbytes()};
    }
    std::vector<int> strides(shape.size(), 0);
    strides[strides.size() - 1] = this->getDataTypeInfo()->nbytes();
    for (int i = shape.size() - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

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

  static std::string
  npOutDataTypeExceptionMessage(const TensorInfo &i0,
                                const TensorInfo &i1,
                                const std::string &debugName);

private:
  const DataTypeInfo *dataTypeInfo = nullptr;
  // The tensor's actual shape
  Shape shape_v;
  // The tensor's meta shape, e.g. original shape before replicated tensor
  // sharding
  Shape meta_shape_v;
};

std::ostream &operator<<(std::ostream &stream, const TensorInfo &ti);
std::ostream &operator<<(std::ostream &stream, const DataType &dt);

} // namespace popart

#endif
