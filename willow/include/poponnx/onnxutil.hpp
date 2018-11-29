#ifndef GUARD_NEURALNET_ONNXUTIL_HPP
#define GUARD_NEURALNET_ONNXUTIL_HPP

#include <onnx/onnx_pb.h>
#include <poponnx/tensordata.hpp>

namespace poponnx {
namespace onnxutil {

// for many types (float16, float, int, etc) onnx::TensorProto has
// 2 ways of storing the data: either in field raw_data or a field
// specific to the type. These functions handle these 2 possibilities.
ConstVoidData getConstData(const onnx::TensorProto &tp);
MutableVoidData getMutableData(onnx::TensorProto &tp);

// Get an ONNX model protobuf, either from a file, or the string directly
onnx::ModelProto getModelProto(const std::string &modelProtoOrFilename);

} // namespace onnxutil
} // namespace poponnx

#endif
