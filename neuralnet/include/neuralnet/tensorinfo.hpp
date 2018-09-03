#ifndef GUARD_NEURALNET_TENSORINFO_HPP
#define GUARD_NEURALNET_TENSORINFO_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <vector>
#include <sstream>
#include <neuralnet/names.hpp>

namespace neuralnet {

class TensorInfo {
  public:
    TensorInfo(DataType, const std::vector<int64_t> &);
    TensorInfo(const onnx::TensorProto & );
    TensorInfo() = default;
    void set(DataType, const std::vector<int64_t> &);
    const std::vector<int64_t> & shape();
    int rank();
    int64_t nelms();
    int64_t nbytes();
    int64_t dim(int i);
    DataType type();
    void append(std::stringstream & ss);

  private:
   DataType type_;
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
