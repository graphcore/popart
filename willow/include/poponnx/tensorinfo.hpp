#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#include <sstream>
#include <vector>
#include <poponnx/names.hpp>

namespace poponnx {

// There is a 1-1 correspondence
// between poponnx::DataTypes
// and onnx::TensorProto_DataTypes, aka
// decltype(onnx::TensorProto().data_type()).

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

class TensorInfo;

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

class TensorInfo {
public:
  TensorInfo(DataType, const Shape &);
  TensorInfo(std::string data_type, std::string shape);
  TensorInfo(std::string data_type, const Shape &);
  explicit TensorInfo(const onnx::TensorProto &);
  explicit TensorInfo(const onnx::TypeProto &);
  void set(const onnx::TensorProto &);
  void set(const onnx::TypeProto &);
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
  DataType dataTypeFromString(const std::string &s) const;
  Shape shapeFromString(const std::string &s) const;
  onnx::TypeProto getOnnxTypeProto() const;
  const DataTypeInfo *getDataTypeInfo() const;

private:
  const DataTypeInfo *dataTypeInfo = nullptr;
  Shape shape_v;
};

std::ostream &operator<<(std::ostream &stream, const TensorInfo &ti);

} // namespace poponnx

#endif
