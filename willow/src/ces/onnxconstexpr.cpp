#include <onnx/onnx_pb.h>

#include <poponnx/ces/onnxconstexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

bool OnnxConstExprUtil::isConst(const onnx::NodeProto &node) {
  return node.op_type() == "Constant" || node.op_type() == "Shape";
}

void OnnxConstExprUtil::processNode(const onnx::NodeProto &node, Ir *ir) {
  if (node.op_type() == "Constant") {
    processConstantNode(node, ir);
  } else if (node.op_type() == "Shape") {
    processShapeNode(node, ir);
  } else {
    throw error("Can not process node type {} as const", node.op_type());
  }
}

void OnnxConstExprUtil::processConstantNode(const onnx::NodeProto &node,
                                            Ir *ir) {
  TensorId name = node.output(0);
  // We assume that a tensor coming from a Constant Node should
  // not have a gradient computed for it or be updated during training
  // We will implement a seperate tool to convert between
  // Constant Operator output and initializer in ONNX models (T6213)
  ir->getTensors().addConstInit(name, &node.attribute(0).t());
}

void OnnxConstExprUtil::processShapeNode(const onnx::NodeProto &node, Ir *ir) {
  std::vector<char> data;
  auto input_id = node.input(0);
  if (!ir->getTensors().contains(input_id)) {
    throw error("No Tensor {} in processShapeNode");
  }
  auto input  = ir->getTensors().get(input_id);
  auto &shape = input->info.shape();

  data.resize(shape.size() * sizeof(int64_t));
  int64_t *int_data = reinterpret_cast<int64_t *>(data.data());

  for (int i = 0; i < shape.size(); i++) {
    int_data[i] = shape[i];
  }

  TensorInfo outInfo(DataType::INT64, {static_cast<int64_t>(shape.size())});
  ir->getTensors().addConstInit(node.output(0), outInfo, data.data());
}

} // namespace poponnx
