// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_GRAPHTRANSFORMER_IMPL_H
#define GUARD_GRAPHTRANSFORMER_IMPL_H

#include <vector>

#include <popart/names.hpp>

#include <onnx/onnx_pb.h>

namespace popart {

class GraphTransformerImpl {
public:
  GraphTransformerImpl(const std::string &modelProtoOrFilename);

  std::string getModelProto() const;

  void convertFloatsToHalfs();
  void convertUINT8ToINT32();
  void convertUINT16ToINT32();
  void convertINT8ToINT32();
  void convertINT16ToINT32();
  void convertINT64ToINT32();
  void convertDoublesToFloats();
  void convertDoublesToHalfs();
  void convertBFloats16ToFloat32();
  void convertInitializersToConstants(const std::vector<TensorId> &ids);
  void convertAllFixedPointInitializersToConstants();
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);
  void prepareNodesForTraining();
  void removeUnusedInputs();

private:
  ONNX_NAMESPACE::ModelProto model;

  static void convertFloatTensorToHalf(ONNX_NAMESPACE::TensorProto &tp);
  static void convertUINT8TensorToINT32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertUINT16TensorToINT32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertINT8TensorToINT32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertINT16TensorToINT32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertINT64TensorToINT32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertDoubleTensorToFloat(ONNX_NAMESPACE::TensorProto &tp);
  static void convertBFloat16TensorToFloat32(ONNX_NAMESPACE::TensorProto &tp);
  static void convertDoubleTensorToHalf(ONNX_NAMESPACE::TensorProto &tp);
};

} // namespace popart
#endif // GUARD_GRAPHTRANSFORMER_H
