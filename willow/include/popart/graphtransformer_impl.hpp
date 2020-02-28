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
  void convertInitializersToConstants(const std::vector<TensorId> &ids);
  void convertAllFixedPointInitializersToConstants();
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);
  void prepareNodesForTraining();
  void removeUnusedInputs();

private:
  onnx::ModelProto model;

  static void convertFloatTensorToHalf(onnx::TensorProto &tp);
};

} // namespace popart
#endif // GUARD_GRAPHTRANSFORMER_H
