#ifndef GUARD_NEURALNET_ONNXUTIL_HPP
#define GUARD_NEURALNET_ONNXUTIL_HPP

#include <onnx/onnx_pb.h>
#include <popart/tensordata.hpp>

namespace popart {
namespace onnxutil {

class ExternalTensorProtoInfo {
public:
  std::string location = "";
  int offset           = 0;
  int length           = 0;

  ExternalTensorProtoInfo(const onnx::TensorProto &tp);
};

// for many types (float16, float, int, etc) onnx::TensorProto has
// 2 ways of storing the data: either in field raw_data or a field
// specific to the type. These functions handle these 2 possibilities.
ConstVoidData getConstData(const onnx::TensorProto &tp);
MutableVoidData getMutableData(onnx::TensorProto &tp);

// Move tensor data for ids from inside ModelProto to external file, fn
void saveInitializersExternally(onnx::ModelProto &model,
                                const std::vector<TensorId> &ids,
                                const std::string &fn);

// Get an ONNX model protobuf, either from a file, or the string directly
onnx::ModelProto getModelProto(const std::string &modelProtoOrFilename);

// From a specified ModelProto, get a TensorProto by its name
onnx::TensorProto getTensorProto(const onnx::ModelProto &model,
                                 const TensorId &tId);
onnx::TensorProto &getTensorProto(onnx::ModelProto &model, const TensorId &tId);

// Is the tensor in the model's initializer list?
bool isInitializer(const onnx::ModelProto &model, const TensorId tId);

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
