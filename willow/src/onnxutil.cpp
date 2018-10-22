#include <willow/error.hpp>
#include <willow/onnxutil.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {
namespace onnxutil {

const void *getData(const onnx::TensorProto &tp) {
  // raw_data is a string: guaranteed to be contiguous in C++11
  if (tp.has_raw_data()) {
    return reinterpret_cast<const void *>(&(tp.raw_data()[0]));
  }
  // note protobuf repeated feld is essentially stl vector (has a function
  // Capactity()) and so data is contiguous
  else {
    if (tp.data_type() == onnx::TensorProto::FLOAT) {
      return reinterpret_cast<const void *>(&(tp.float_data().Get(0)));
    } else if (tp.data_type() == onnx::TensorProto::INT64) {
      return reinterpret_cast<const void *>(&(tp.int64_data().Get(0)));
    }

    else {
      TensorInfo tInfo(tp);
      throw error("getData need implementing for" + tInfo.data_type());
    }
  }
}

} // namespace onnxutil
} // namespace willow
