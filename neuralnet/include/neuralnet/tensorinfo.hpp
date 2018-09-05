#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <neuralnet/names.hpp>
#include <sstream>
#include <vector>

namespace neuralnet {

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

class TensorInfo {
public:
  TensorInfo(DataType, const std::vector<int64_t> &);
  TensorInfo(const onnx::TensorProto &);
  void set(const onnx::TensorProto &);
  TensorInfo() = default;
  void set(DataType, const std::vector<int64_t> &);
  const std::vector<int64_t> &shape() const;
  int rank() const;
  int64_t nelms() const;
  int64_t nbytes() const;
  int64_t dim(int i) const;
  DataType dataType() const;
  const std::string & data_type() const;
  void append(std::stringstream &) const;
  bool isSet() const;

private:
  const DataTypeInfo *dataTypeInfo = nullptr;
  std::vector<int64_t> shape_v;
};

template <class T> void appendSequence(std::stringstream &ss, T t) {
  int index = 0;
  ss << '[';
  for (auto &x : t) {
    if (index != 0) {
      ss << ' ';
    }
    ss << x;
    ++index;
  }
  ss << ']';
}

} // namespace neuralnet

#endif
