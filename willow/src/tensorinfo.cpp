#include <algorithm>
#include <onnx/onnx_pb.h>
#include <popart/error.hpp>
#include <popart/onnxutil.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>

namespace popart {

template <> DataType getDataType<int8_t>() { return DataType::INT8; }
template <> DataType getDataType<int16_t>() { return DataType::INT16; }
template <> DataType getDataType<int32_t>() { return DataType::INT32; }
template <> DataType getDataType<int64_t>() { return DataType::INT64; }
template <> DataType getDataType<uint8_t>() { return DataType::UINT8; }
template <> DataType getDataType<uint16_t>() { return DataType::UINT16; }
template <> DataType getDataType<uint32_t>() { return DataType::UINT32; }
template <> DataType getDataType<uint64_t>() { return DataType::UINT64; }
template <> DataType getDataType<bool>() { return DataType::BOOL; }
template <> DataType getDataType<Half>() { return DataType::FLOAT16; }
template <> DataType getDataType<float>() { return DataType::FLOAT; }
template <> DataType getDataType<double>() { return DataType::DOUBLE; }
template <> DataType getDataType<std::string>() { return DataType::STRING; }

TensorInfo::TensorInfo(DataType t, const Shape &s)
    : dataTypeInfo(&getDataTypeInfoMap().at(t)), shape_v(s) {}

TensorInfo::TensorInfo(std::string s_type, const Shape &s)
    : TensorInfo(dataTypeFromString(s_type), s) {}

TensorInfo::TensorInfo(const onnx::TensorProto &t) { set(t); }

TensorInfo::TensorInfo(const onnx::TypeProto &t) { set(t); }

TensorInfo::TensorInfo(std::string s_type, std::string s_shape)
    : TensorInfo(dataTypeFromString(s_type), shapeFromString(s_shape)) {}

void TensorInfo::set(const onnx::TensorProto &t) {
  dataTypeInfo = &getDataTypeInfoMap().at(onnxutil::getDataType(t.data_type()));
  shape_v.clear();
  for (auto &v : t.dims()) {
    shape_v.push_back(v);
  }
  shape_v.shrink_to_fit();
}

void TensorInfo::set(const onnx::TypeProto &t) {
  auto type = t.tensor_type();
  dataTypeInfo =
      &getDataTypeInfoMap().at(onnxutil::getDataType(type.elem_type()));
  shape_v.clear();
  for (auto &v : type.shape().dim()) {
    if (v.has_dim_param()) {
      throw error("Tensor shape requires unspecified parameter '{}'",
                  v.dim_param());
    } else {
      shape_v.push_back(v.dim_value());
    }
  }
  shape_v.shrink_to_fit();
}

std::vector<size_t> TensorInfo::shape_szt() const {
  std::vector<size_t> szts;
  szts.reserve(rank());
  for (auto &x : shape()) {
    szts.push_back(static_cast<size_t>(x));
  }
  return szts;
}

static bool isBroadcastableDims(int64_t a, int64_t b) {
  if ((a > 0) && (b > 0) && ((a == b) || (a == 1) || (b == 1))) {
    return true;
  } else {
    return false;
  }
}

static int64_t broadcastableDimSize(int64_t a, int64_t b) {
  if (isBroadcastableDims(a, b)) {
    return std::max(a, b);
  } else {
    // Incompatible dimensions found. Throw an exception,
    // borrowing the same terminology as numpy.
    throw error("np broadcasting failed, frames are not aligned");
  }
}

static std::string npOutExceptionMessage(const std::vector<int64_t> &s0,
                                         const std::vector<int64_t> &s1,
                                         const std::string &debugName) {
  std::stringstream ss;

  const auto reduction = [](std::string a, int64_t b) {
    return a + (a.empty() ? "" : ", ") + std::to_string(b);
  };

  const auto s0_str =
      std::accumulate(s0.begin(), s0.end(), std::string{}, reduction);
  const auto s1_str =
      std::accumulate(s1.begin(), s1.end(), std::string{}, reduction);

  ss << "np broadcasting failed";

  if (!debugName.empty()) {
    ss << " on '" << debugName << '\'';
  }

  ss << ", frames [" << s0_str << "] and [" << s1_str << "] are not aligned";

  return ss.str();
}

// Calculate the numpy broadcast shape as described in
// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
//
// For Example:
// s0            = {   1, 4, 5} &
// s1            = {2, 3, 1, 1} =>
// npOut(s0, s1) = {2, 3, 4, 5}
// Look in tests/popart/numpybroadcastshapetest.cpp for more examples.
std::vector<int64_t> npOut(const std::vector<int64_t> &s0,
                           const std::vector<int64_t> &s1,
                           const std::string &debugName) {
  // In the given example:
  // s0      = {   1, 4, 5} &
  // s1      = {2, 3, 4, 5} =>
  // result := {1, 1, 1, 1}
  std::vector<int64_t> result(std::max(s0.size(), s1.size()), 1);

  // Calculate the length of the overlap between `s0` and `s1` in `result`.
  //
  // In the given example:
  // overlap := min(|s0|, |s1|) = 3
  const auto overlap = std::min(s0.size(), s1.size());

  // If the lengths of `s0` and `s1` mismatch, copy the prefix of the longer
  // shape into `result`. At least one of these copies will be a no-op.
  //
  // In the given example:
  // overlap = 3            &
  // s0      = {   1, 4, 5} &
  // s1      = {2, 3, 1, 1} &
  // result  = {1, 1, 1, 1} =>
  // result := {2, 1, 1, 1}
  const auto dst_itr = std::next(result.rbegin(), overlap);
  std::copy(std::next(s0.rbegin(), overlap), s0.rend(), dst_itr);
  std::copy(std::next(s1.rbegin(), overlap), s1.rend(), dst_itr);

  // Check that the overlapping region is numpy broadcastable.
  if (!std::inner_product(s0.rbegin(),
                          std::next(s0.rbegin(), overlap),
                          s1.rbegin(),
                          true,
                          std::logical_and<bool>(),
                          isBroadcastableDims)) {
    throw error(npOutExceptionMessage(s0, s1, debugName));
  }

  // Take the element-wise maximum of `s0` and `s1` in the overlapping region
  // and put it into `result`. This will throw an exception if the elements are
  // not numpy broadcast compatible.
  //
  // In the given example:
  // overlap = 3            &
  // s0      = {   1, 4, 5} &
  // s1      = {2, 3, 1, 1} &
  // result  = {2, 1, 1, 1} =>
  // result := {2, 3, 4, 5}
  util::zipWith(s0.rbegin(),
                std::next(s0.rbegin(), overlap),
                s1.rbegin(),
                std::next(s1.rbegin(), overlap),
                result.rbegin(),
                broadcastableDimSize);

  return result;
}

TensorInfo npOut(const TensorInfo &i0, const TensorInfo &i1) {
  if (i0.dataType() != i1.dataType()) {
    throw error(("np broadcasting failed, incompatible types {} and {} "
                 "(shapes {} and {})"),
                i0.data_type(),
                i1.data_type(),
                i0.shape(),
                i1.shape());
  }

  return {i0.dataType(), npOut(i0.shape(), i1.shape())};
}

// Compute the reduction axis for a reduction op.
//
// For Example:
// in            = {   1, 4, 5} &
// out           = {2, 3, 4, 5} =>
// npIn(in, out) = {0, 1}
std::vector<int64_t> npReductionAxis(const std::vector<int64_t> &in,
                                     const std::vector<int64_t> &out) {
  // Calculate the length of the overlap between `in` and `out`, the length of
  // the prefix, and the number of differences between `in` and `out`.
  //
  // In the given example:
  // in       = {   1, 4, 5}          &
  // out      = {2, 3, 4, 5}          =>
  // overlap := |in|              = 3 &
  // prefix  := |out|  - overlap  = 1 &
  // diffs   := mismatch(in, out) = 2
  const auto overlap = in.size();
  const auto prefix  = out.size() - overlap;
  const auto diffs =
      util::count_mismatch(in.rbegin(), in.rend(), out.rbegin(), out.rend());

  // Create a vector for storing the axes to reduce over. This will be equal to
  // the number of differences between `in` and `out`.
  //
  // In the given example:
  // diffs = 2 =>
  // axes := {*, *}
  std::vector<int64_t> axes(diffs);

  // The prefix axis must be included. If `in` and `out` have equal rank, this
  // will be a no-op.
  //
  // In the given example:
  // in     = {   1, 4, 5} &
  // out    = {2, 3, 4, 5} &
  // prefix = 1            &
  // axes   = {*, *}       =>
  // axes  := {0, *}
  std::iota(axes.begin(), std::next(axes.begin(), prefix), 0);

  // For the remaining axes, find the mismatches in the overlapping region and
  // put the indices in `axes`. We are guaranteed to find exactly `diffs -
  // prefix` mismatches.
  //
  // In the given example:
  // in      = {   1, 4, 5} &
  // out     = {2, 3, 4, 5} &
  // diffs   = 2            &
  // prefix  = 1            &
  // axes    = {0, *}       =>
  // axes   := {0, 1}
  auto itr_o = std::next(axes.begin(), prefix);
  auto itr_a = in.begin();
  auto itr_b = std::next(out.begin(), prefix);
  for (int i = 0; i < (diffs - prefix); ++i) {
    std::tie(itr_a, itr_b) = std::mismatch(itr_a, in.end(), itr_b);
    *itr_o                 = prefix + std::distance(in.begin(), itr_a);

    itr_o++;
    itr_a++;
    itr_b++;
  }

  // Return the axes.
  //
  // In the given example:
  // axes = {0, 1}
  return axes;
}

void TensorInfo::append(std::ostream &ss) const {
  ss << padded(dataTypeInfo->name(), 8);
  appendSequence(ss, shape_v);
}

bool TensorInfo::isSet() const { return dataTypeInfo != nullptr; }

const std::string &TensorInfo::data_type() const {
  return dataTypeInfo->name();
}

const std::string &TensorInfo::data_type_lcase() const {
  return dataTypeInfo->lcasename();
}

const Shape &TensorInfo::shape() const { return shape_v; }

Rank TensorInfo::rank() const { return static_cast<int>(shape_v.size()); }

int64_t TensorInfo::nelms() const {
  return std::accumulate(shape_v.begin(),
                         shape_v.end(),
                         static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

int64_t TensorInfo::nbytes() const {
  return nelms() * static_cast<int64_t>(dataTypeInfo->nbytes());
}

int64_t TensorInfo::dim(int i) const { return shape_v[i]; }

DataType TensorInfo::dataType() const { return dataTypeInfo->type(); }

void TensorInfo::set(DataType t, const Shape &s) {
  dataTypeInfo = &getDataTypeInfoMap().at(t);
  shape_v      = s;
}

const std::map<DataType, DataTypeInfo> &getDataTypeInfoMap() {
  static std::map<DataType, DataTypeInfo> dataTypeInfoMap =
      initDataTypeInfoMap();
  return dataTypeInfoMap;
}

bool TensorInfo::operator==(const TensorInfo &i1) const {
  return (shape_v == i1.shape_v && dataTypeInfo == i1.dataTypeInfo);
}

bool TensorInfo::operator!=(const TensorInfo &i1) const {
  return !(operator==(i1));
}

std::map<DataType, DataTypeInfo> initDataTypeInfoMap() {

  return {
      {DataType::UNDEFINED,
       {DataType::UNDEFINED, -1, false, "UNDEFINED", "undefined"}},
      {DataType::FLOAT, {DataType::FLOAT, 4, false, "FLOAT", "float32"}},
      {DataType::UINT8, {DataType::UINT8, 1, true, "UINT8", "uint"}},
      {DataType::INT8, {DataType::INT8, 1, true, "INT8", "int8"}},
      {DataType::UINT16, {DataType::UINT16, 2, true, "UINT16", "uint16"}},
      {DataType::INT16, {DataType::INT16, 2, true, "INT16", "int16"}},
      {DataType::INT32, {DataType::INT32, 4, true, "INT32", "int32"}},
      {DataType::INT64, {DataType::INT64, 8, true, "INT64", "int64"}},
      {DataType::STRING, {DataType::STRING, -1, false, "STRING", "string"}},
      {DataType::BOOL, {DataType::BOOL, 1, true, "BOOL", "bool"}},
      {DataType::FLOAT16, {DataType::FLOAT16, 2, false, "FLOAT16", "float16"}},
      {DataType::BFLOAT16,
       {DataType::BFLOAT16, 2, false, "BFLOAT16", "bfloat16"}},
      {DataType::DOUBLE, {DataType::DOUBLE, 8, false, "DOUBLE", "float64"}},
      {DataType::UINT32, {DataType::UINT32, 4, false, "UINT32", "uint32"}},
      {DataType::UINT64, {DataType::UINT64, 8, false, "UINT64", "uint64"}},
      {DataType::COMPLEX64,
       {DataType::COMPLEX64, 8, false, "COMPLEX64", "complex64"}},
      {DataType::COMPLEX128,
       {DataType::COMPLEX128, 16, false, "COMPLEX128", "complex128"}}};
}

std::map<std::string, DataType> initStrToDataTypeMap() {
  std::map<std::string, DataType> invMap;
  for (auto &type_inf : getDataTypeInfoMap()) {
    auto dtInf           = type_inf.second;
    invMap[dtInf.name()] = dtInf.type();
  }
  return invMap;
}

const std::string &getAllONNXTypesString() {
  const static std::string allTypes = initAllONNXTypesString();
  return allTypes;
}

std::string initAllONNXTypesString() {
  std::stringstream allTypes;
  allTypes << '[';
  bool firstType = true;
  for (auto &name_type : getStrToDataTypeMap()) {
    if (firstType) {
      firstType = false;
    } else {
      allTypes << ',' << ' ';
    }
    allTypes << name_type.first;
  }
  allTypes << ']';
  return allTypes.str();
}

DataType dataTypeFromString(const std::string &s) {
  auto found = getStrToDataTypeMap().find(s);
  if (found == getStrToDataTypeMap().end()) {
    throw error("no ONNX type {}, they're {}.", s, getAllONNXTypesString());
  }
  return found->second;
}

// expects shape to be "(1 2 400 3)" or "(5)", so no spaces allowed.
Shape TensorInfo::shapeFromString(const std::string &s) const {
  if (s.size() < 2 || s[0] != '(' || s[s.size() - 1] != ')') {
    throw error("invalid string for shape");
  }
  if (s.find(' ') != std::string::npos) {
    throw error("s contains a space : not valid shape string");
  }

  Shape shape;

  // https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
  std::string token;
  std::istringstream tokenStream(s.substr(1, s.size() - 2));
  while (std::getline(tokenStream, token, ',')) {
    shape.push_back(std::stoi(token));
  }

  std::stringstream ss;
  return shape;
}

onnx::TypeProto TensorInfo::getOnnxTypeProto() const {
  onnx::TypeProto typeProto;

  onnx::TypeProto_Tensor *tensor = typeProto.mutable_tensor_type();
  tensor->set_elem_type(onnxutil::getTPDataType(dataTypeInfo->type()));

  onnx::TensorShapeProto *shape = tensor->mutable_shape();
  for (auto d : shape_v) {
    shape->add_dim()->set_dim_value(d);
  }

  return typeProto;
}

const DataTypeInfo *TensorInfo::getDataTypeInfo() const { return dataTypeInfo; }

const std::map<std::string, DataType> &getStrToDataTypeMap() {
  static std::map<std::string, DataType> m = initStrToDataTypeMap();
  return m;
}

std::ostream &operator<<(std::ostream &stream, const TensorInfo &ti) {
  ti.append(stream);
  return stream;
}

std::ostream &operator<<(std::ostream &stream, const DataType &dt) {
  stream << getDataTypeInfoMap().at(dt).lcasename();
  return stream;
}

DataTypeInfo::DataTypeInfo(DataType type__,
                           int nbytes__,
                           bool isFixedPoint__,
                           std::string name__,
                           std::string lcasename__)
    : type_(type__), nbytes_(nbytes__), isFixedPoint_(isFixedPoint__),
      name_(name__), lcasename_(lcasename__) {}

const int &DataTypeInfo::nbytes() const { return nbytes_; }

const std::string &DataTypeInfo::name() const { return name_; }

const std::string &DataTypeInfo::lcasename() const { return lcasename_; }

bool DataTypeInfo::isFixedPoint() const { return isFixedPoint_; }

DataType DataTypeInfo::type() const { return type_; }

} // namespace popart
