#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx_pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <sstream>
#include <vector>
#include <willow/names.hpp>

namespace willow {

class TensorInfo;

// numpy output shape of:
std::vector<int64_t> npOut(const std::vector<int64_t> &s0,
                           const std::vector<int64_t> &s1);

TensorInfo npOut(const TensorInfo &i0, const TensorInfo &i1);

// FLOAT, FLOAT16, INT8 etc.
class DataTypeInfo {
public:
  DataTypeInfo(DataType type__, int nbytes__, std::string name__);
  DataType type() const;
  // number of bytes of 1 element
  const int &nbytes() const;
  const std::string &name() const;

private:
  DataType type_;
  int nbytes_;
  std::string name_;
};

const std::map<DataType, DataTypeInfo> &getDataTypeInfoMap();
std::map<DataType, DataTypeInfo> initDataTypeInfoMap();

const std::map<std::string, DataType> &getStrToDataTypeMap();
std::map<std::string, DataType> initStrToDataTypeMap();

const std::string &getAllONNXTypesString();
std::string initAllONNXTypesString();

class TensorInfo {
public:
  TensorInfo(DataType, const std::vector<int64_t> &);
  TensorInfo(std::string data_type, std::string shape);
  TensorInfo(std::string data_type, const std::vector<int64_t> &);
  TensorInfo(const onnx::TensorProto &);
  void set(const onnx::TensorProto &);
  TensorInfo() = default;
  void set(DataType, const std::vector<int64_t> &);
  const std::vector<int64_t> &shape() const;
  int rank() const;
  int64_t nelms() const;
  // total bytes of tensor
  int64_t nbytes() const;
  int64_t dim(int i) const;
  DataType dataType() const;
  const std::string &data_type() const;
  void append(std::stringstream &) const;
  bool isSet() const;
  bool operator==(const TensorInfo &) const;
  bool operator!=(const TensorInfo &) const;
  DataType dataTypeFromString(const std::string &s) const;
  std::vector<int64_t> shapeFromString(const std::string &s) const;

private:
  const DataTypeInfo *dataTypeInfo = nullptr;
  std::vector<int64_t> shape_v;
};

} // namespace willow

#endif
