// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
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

  ExternalTensorProtoInfo(const ONNX_NAMESPACE::TensorProto &tp);
};

// for many types (float16, float, int, etc) ONNX_NAMESPACE::TensorProto has
// 2 ways of storing the data: either in field raw_data or a field
// specific to the type. These functions handle these 2 possibilities.
ConstVoidData getConstData(const ONNX_NAMESPACE::TensorProto &tp);
MutableVoidData getMutableData(ONNX_NAMESPACE::TensorProto &tp);

// Move tensor data for ids from inside ModelProto to external file, fn
void saveInitializersExternally(ONNX_NAMESPACE::ModelProto &model,
                                const std::vector<TensorId> &ids,
                                const std::string &fn);

// Get an ONNX model protobuf, either from a file, or the string directly
ONNX_NAMESPACE::ModelProto
getModelProto(const std::string &modelProtoOrFilename);

// From a specified ModelProto, get a TensorProto by its name
ONNX_NAMESPACE::TensorProto
getTensorProto(const ONNX_NAMESPACE::ModelProto &model, const TensorId &tId);
ONNX_NAMESPACE::TensorProto &getTensorProto(ONNX_NAMESPACE::ModelProto &model,
                                            const TensorId &tId);

// Is the tensor in the model's initializer list?
bool isInitializer(const ONNX_NAMESPACE::ModelProto &model, const TensorId tId);

// functions for translating between popart's enum class and onnx's enum
ONNX_NAMESPACE::TensorProto_DataType getTPDataType(DataType);
DataType getDataType(int);

void visitModelNodes(ONNX_NAMESPACE::ModelProto &model,
                     std::function<void(ONNX_NAMESPACE::NodeProto &)> f);
void visitModelInitializers(
    ONNX_NAMESPACE::ModelProto &model,
    std::function<void(ONNX_NAMESPACE::TensorProto &)> f);
void visitModelValueInfos(
    ONNX_NAMESPACE::ModelProto &model,
    std::function<void(ONNX_NAMESPACE::ValueInfoProto &)> f);

} // namespace onnxutil
} // namespace popart

#endif
