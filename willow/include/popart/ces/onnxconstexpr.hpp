// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXCONSTEXPR_HPP
#define GUARD_NEURALNET_ONNXCONSTEXPR_HPP

namespace popart {

class Graph;

class OnnxConstExprUtil {
public:
  static bool isConst(const ONNX_NAMESPACE::NodeProto &);
  static void processNode(const ONNX_NAMESPACE::NodeProto &, Graph *);

private:
  static void processConstantNode(const ONNX_NAMESPACE::NodeProto &, Graph *);
  static void processShapeNode(const ONNX_NAMESPACE::NodeProto &, Graph *);
  static void processConstantOfShapeNode(const ONNX_NAMESPACE::NodeProto &,
                                         Graph *);
};

} // namespace popart

#endif
