#ifndef GUARD_GRAPHTRANSFORMER_H
#define GUARD_GRAPHTRANSFORMER_H

#include <memory>
#include <string>
#include <vector>

#include <poponnx/names.hpp>

namespace poponnx {

class GraphTransformerImpl;

class GraphTransformer {
public:
  GraphTransformer(const std::string &modelProtoOrFilename);
  ~GraphTransformer();

  std::string getModelProto() const;

  /**
   * Convert the graph from float32 to float16
   */
  void convertFloatsToHalfs();

  /**
   * Convert the given list of initializers into ONNX Constant Nodes
   *
   * \param ids A list of initializer names
   */
  void convertInitializersToConstants(const std::vector<TensorId> &ids);

  /**
   * Convert all of the fixed-point initializers into ONNX Constant Nodes
   */
  void convertAllFixedPointInitializersToConstants();

private:
  std::unique_ptr<GraphTransformerImpl> impl;
};

} // namespace poponnx
#endif // GUARD_GRAPHTRANSFORMER_H
