#ifndef GUARD_NEURALNET_ONNXUTIL_HPP
#define GUARD_NEURALNET_ONNXUTIL_HPP

#include <onnx/onnx_pb.h>
#include <popart/tensordata.hpp>

namespace popart {
namespace onnxutil {

// for many types (float16, float, int, etc) onnx::TensorProto has
// 2 ways of storing the data: either in field raw_data or a field
// specific to the type. These functions handle these 2 possibilities.
ConstVoidData getConstData(const onnx::TensorProto &tp);
MutableVoidData getMutableData(onnx::TensorProto &tp);

// Get an ONNX model protobuf, either from a file, or the string directly
onnx::ModelProto getModelProto(const std::string &modelProtoOrFilename);

// From a specified ModelProto, get a TensorProto by its name
onnx::TensorProto getTensorProto(const onnx::ModelProto &model,
                                 const TensorId tId);

// functions for translating between popart's enum class and onnx's enum
onnx::TensorProto_DataType getTPDataType(DataType);
DataType getDataType(int);

void visitModelNodes(onnx::ModelProto &model,
                     std::function<void(onnx::NodeProto &)> f);
void visitModelInitializers(onnx::ModelProto &model,
                            std::function<void(onnx::TensorProto &)> f);
void visitModelValueInfos(onnx::ModelProto &model,
                          std::function<void(onnx::ValueInfoProto &)> f);

} // namespace onnxutil
} // namespace popart

#endif
