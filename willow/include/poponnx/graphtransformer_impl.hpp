#ifndef GUARD_GRAPHTRANSFORMER_IMPL_H
#define GUARD_GRAPHTRANSFORMER_IMPL_H

#include <vector>

#include <poponnx/names.hpp>

#include <onnx/onnx_pb.h>

namespace poponnx {

class GraphTransformerImpl {
public:
  GraphTransformerImpl(const std::string &modelProtoOrFilename);

  std::string getModelProto() const;

  void convertFloatsToHalfs();
  void convertInitializersToConstants(const std::vector<TensorId> &ids);
  void convertAllFixedPointInitializersToConstants();
  void prepareNodesForTraining();
  void removeUnusedInputs();

private:
  onnx::ModelProto model;

  static void convertFloatTensorToHalf(onnx::TensorProto &tp);
};

} // namespace poponnx
#endif // GUARD_GRAPHTRANSFORMER_H
