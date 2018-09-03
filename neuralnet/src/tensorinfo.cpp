#include <neuralnet/error.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <numeric>

namespace neuralnet {

TensorInfo::TensorInfo(DataType t, const std::vector<int64_t> &s)
    : type_(t), shape_v(s) {}

TensorInfo::TensorInfo(const onnx::TensorProto &t) : type_(t.data_type()) {
  shape_v.reserve(static_cast<uint64_t>(t.dims_size()));
  for (auto &v : t.dims()) {
    shape_v.push_back(v);
  }
}

void TensorInfo::append(std::stringstream & ss){
  ss << "shape: ";
  appendSequence(ss, shape_v);
  ss << " type: " << type_;
}

const std::vector<int64_t> &TensorInfo::shape() { return shape_v; }

int TensorInfo::rank() { return static_cast<int>(shape_v.size()); }

int64_t TensorInfo::nelms() {
  return std::accumulate(
      shape_v.begin(), shape_v.end(), 1, std::multiplies<int64_t>());
}

int64_t TensorInfo::dim(int i) { return shape_v[static_cast<uint>(i)]; }

DataType TensorInfo::type() { return type_; }

void TensorInfo::set(DataType t, const std::vector<int64_t> & s){
  type_ = t;
  shape_v = s;
}

int64_t TensorInfo::nbytes(){
  throw error("nbytes not implemented");

}

} // namespace neuralnet
