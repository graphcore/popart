// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_GRAPHTRANSFORMER_H
#define GUARD_GRAPHTRANSFORMER_H

#include <memory>
#include <string>
#include <vector>

#include <popart/names.hpp>

namespace popart {

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

  /**
   * Some ONNX Operators are different between train and test modes
   * An example is BatchNormalization, which has 1 output in test mode
   * and 5 outputs in train mode
   * This function changes the Nodes to be of the training variety
   */
  void prepareNodesForTraining();

  /**
   * Inputs which are not connected to any Node are removed
   */
  void removeUnusedInputs();

  /**
   * The model data cannot exceed 2GB - the maximum size of a Protobuf
   * message. To prevent this for large models, ONNX tensor data can be
   * saved separately.
   *
   * \param ids The names of tensors whose data is to be saved externally.
   * \param fn The name of a file containing the binary tensor data.
   */
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);

private:
  std::unique_ptr<GraphTransformerImpl> impl;
};

} // namespace popart
#endif // GUARD_GRAPHTRANSFORMER_H
