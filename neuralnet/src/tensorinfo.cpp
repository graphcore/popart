#include <neuralnet/error.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <numeric>

namespace neuralnet {

TensorInfo::TensorInfo(DataType t, const std::vector<int64_t> &s)
    : dataTypeInfo(&getDataTypeInfoMap().at(t)), shape_v(s) {}

TensorInfo::TensorInfo(const onnx::TensorProto &t) { set(t); }

void TensorInfo::set(const onnx::TensorProto & t){
  dataTypeInfo = &getDataTypeInfoMap().at(t.data_type());
  shape_v.reserve(static_cast<uint64_t>(t.dims_size()));
  for (auto &v : t.dims()) {
    shape_v.push_back(v);
  }
}


void TensorInfo::append(std::stringstream &ss) const {
  ss << "shape: ";
  appendSequence(ss, shape_v);
  ss << " type: " << dataTypeInfo->name();
}

bool TensorInfo::isSet() const{
  return dataTypeInfo != nullptr;
}

const std::string &TensorInfo::data_type() const {
  return dataTypeInfo->name();
}

const std::vector<int64_t> &TensorInfo::shape() const { return shape_v; }

int TensorInfo::rank() const { return static_cast<int>(shape_v.size()); }

int64_t TensorInfo::nelms() const {
  return std::accumulate(
      shape_v.begin(), shape_v.end(), 1, std::multiplies<int64_t>());
}

int64_t TensorInfo::nbytes() const {
  return nelms() * static_cast<int64_t>(dataTypeInfo->nbytes());
}

int64_t TensorInfo::dim(int i) const { return shape_v[static_cast<uint>(i)]; }

DataType TensorInfo::dataType() const { return dataTypeInfo->type(); }

void TensorInfo::set(DataType t, const std::vector<int64_t> &s) {
  dataTypeInfo = &getDataTypeInfoMap().at(t);
  shape_v      = s;
}

const std::map<DataType, DataTypeInfo> &getDataTypeInfoMap() {
  static std::map<DataType, DataTypeInfo> dataTypeInfoMap =
      initDataTypeInfoMap();
  return dataTypeInfoMap;
}

std::map<DataType, DataTypeInfo> initDataTypeInfoMap() {

  return {{TP::UNDEFINED, {TP::UNDEFINED, -1, "UNDEFINED"}},
          {TP::FLOAT, {TP::FLOAT, 4, "FLOAT"}},
          {TP::UINT8, {TP::UINT8, 1, "UINT8"}},
          {TP::INT8, {TP::INT8, 1, "INT8"}},
          {TP::UINT16, {TP::UINT16, 2, "UINT16"}},
          {TP::INT16, {TP::INT16, 2, "INT16"}},
          {TP::INT32, {TP::INT32, 4, "INT32"}},
          {TP::INT64, {TP::INT64, 8, "INT64"}},
          {TP::STRING, {TP::STRING, -1, "STRING"}},
          {TP::BOOL, {TP::BOOL, 1, "BOOL"}},
          {TP::FLOAT16, {TP::FLOAT16, 2, "FLOAT16"}},
          {TP::DOUBLE, {TP::DOUBLE, 8, "DOUBLE"}},
          {TP::UINT32, {TP::UINT32, 4, "UINT32"}},
          {TP::UINT64, {TP::UINT64, 8, "UINT64"}},
          {TP::COMPLEX64, {TP::COMPLEX64, 8, "COMPLEX64"}},
          {TP::COMPLEX128, {TP::COMPLEX128, 16, "COMPLEX128"}}};
}

DataTypeInfo::DataTypeInfo(DataType type__, int nbytes__, std::string name__)
    : type_(type__), nbytes_(nbytes__), name_(name__) {}

const int &DataTypeInfo::nbytes() const { return nbytes_; }

const std::string &DataTypeInfo::name() const { return name_; }

DataType DataTypeInfo::type() const { return type_; }

} // namespace neuralnet
