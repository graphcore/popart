#include <willow/error.hpp>
#include <willow/onnxutil.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {
namespace onnxutil {

ConstVoidData getConstData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data;
  cv_data.info = TensorInfo(tp);
  cv_data.data = nullptr;
  // raw_data is a string: guaranteed to be contiguous in C++11
  if (tp.has_raw_data()) {
    cv_data.data = reinterpret_cast<const void *>(&(tp.raw_data()[0]));
  }
  // note protobuf repeated feld is essentially stl vector (has a function
  // Capactity()) and so data is contiguous
  else {
    if (tp.data_type() == onnx::TensorProto::FLOAT) {
      cv_data.data = reinterpret_cast<const void *>(&(tp.float_data().Get(0)));
    } else if (tp.data_type() == onnx::TensorProto::INT64) {
      cv_data.data = reinterpret_cast<const void *>(&(tp.int64_data().Get(0)));
    }

    else {
      throw error("getConstData needs implementing for" +
                  cv_data.info.data_type());
    }
  }

  return cv_data;
}

MutableVoidData getMutableData(onnx::TensorProto &tp) {
  MutableVoidData mv_data;
  mv_data.info = TensorInfo(tp);
  mv_data.data = nullptr;
  if (tp.has_raw_data()) {
    mv_data.data = reinterpret_cast<void *>(&((*tp.mutable_raw_data())[0]));
  } else {
    if (tp.data_type() == onnx::TensorProto::FLOAT) {
      mv_data.data =
          reinterpret_cast<void *>(tp.mutable_float_data()->Mutable(0));
    } else {
      throw error("getMutableData needs implementing for " +
                  mv_data.info.data_type());
    }
  }
  return mv_data;
}

} // namespace onnxutil
} // namespace willow
