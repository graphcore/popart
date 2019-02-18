#ifndef GUARD_NEURALNET_ONNXCONSTEXPR_HPP
#define GUARD_NEURALNET_ONNXCONSTEXPR_HPP

namespace poponnx {

class Ir;

class OnnxConstExprUtil {
public:
  static bool isConst(const onnx::NodeProto &);
  static void processNode(const onnx::NodeProto &, Ir *);

private:
  static void processConstantNode(const onnx::NodeProto &, Ir *);
  static void processShapeNode(const onnx::NodeProto &, Ir *);
};

} // namespace poponnx

#endif
