#ifndef GUARD_NEURALNET_ONNXCONSTEXPR_HPP
#define GUARD_NEURALNET_ONNXCONSTEXPR_HPP

namespace poponnx {

class Graph;

class OnnxConstExprUtil {
public:
  static bool isConst(const onnx::NodeProto &);
  static void processNode(const onnx::NodeProto &, Graph *);

private:
  static void processConstantNode(const onnx::NodeProto &, Graph *);
  static void processShapeNode(const onnx::NodeProto &, Graph *);
};

} // namespace poponnx

#endif
