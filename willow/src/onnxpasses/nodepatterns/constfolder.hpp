// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDER_HPP
#define GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDER_HPP

#include <map>
#include <onnxpasses/nodepattern.hpp>

namespace popart {
namespace onnxpasses {

class ConstFoldOp;

// Constant fold ops if possible and cache results of the folding.
class ConstFolder : public NodePattern {

public:
  ConstFolder(std::shared_ptr<PatternTarget> t);
  ~ConstFolder();

private:
  void registerConstFoldOpMap();

  // Check if all inputs of the NodeProto \a node are available as
  // poprithm host Tensor. If not 0th element of returned tuple is
  // false, otherwise true. The 1st element are the inputs host Tensor.
  std::tuple<bool, Constants> inputReady(const NodeProto &node) const;

  bool go(const NodeProto &node) final;

  // List of constant folding ops.
  std::map<std::string, std::unique_ptr<ConstFoldOp>> constFoldOpMap;

  // Constant Tensors obtained by constant folding of GraphProto.
  std::shared_ptr<Constants> foldConstants;
};

} // namespace onnxpasses
} // namespace popart

#endif
