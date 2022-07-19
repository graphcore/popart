// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_ONNXCONSTEXPR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_ONNXCONSTEXPR_HPP_

namespace ONNX_NAMESPACE {
class NodeProto;
}
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

#endif // POPART_WILLOW_INCLUDE_POPART_CES_ONNXCONSTEXPR_HPP_
