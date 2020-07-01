// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>

#include <popart/ces/onnxconstexpr.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/onnxutil.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

bool OnnxConstExprUtil::isConst(const ONNX_NAMESPACE::NodeProto &node) {
  return node.op_type() == "Constant" || node.op_type() == "Shape" ||
         node.op_type() == "ConstantOfShape";
}

void OnnxConstExprUtil::processNode(const ONNX_NAMESPACE::NodeProto &node,
                                    Graph *graph) {
  if (node.op_type() == "Constant") {
    processConstantNode(node, graph);
  } else if (node.op_type() == "Shape") {
    processShapeNode(node, graph);
  } else if (node.op_type() == "ConstantOfShape") {
    processConstantOfShapeNode(node, graph);
  } else {
    throw error("Cannot process node type {} as const", node.op_type());
  }
}

void OnnxConstExprUtil::processConstantNode(
    const ONNX_NAMESPACE::NodeProto &node,
    Graph *graph) {
  // Add the scope of the current graph to the tensorID of the node output, this
  // is done for ops as standard, but we also need to do it for the special case
  // of constants.
  TensorId name = graph->addScope(node.output(0));
  // We assume that a tensor coming from a Constant Node should
  // not have a gradient computed for it or be updated during training
  // We will implement a separate tool to convert between
  // Constant Operator output and initializer in ONNX models (T6213)

  // Search the constant node for the 'value' attribute, and return the value
  // tensor.
  auto getValueAttribute = [](auto &node) {
    for (auto &attr : node.attribute()) {
      if (attr.name() == "value") {
        return &attr.t();
      }
    }
    throw error("Could not find the 'value' attribute on the constant node");
  };

  graph->getTensors().addConstInit(name, getValueAttribute(node));
}

void OnnxConstExprUtil::processShapeNode(const ONNX_NAMESPACE::NodeProto &node,
                                         Graph *graph) {
  std::vector<char> data;
  auto input_id = node.input(0);
  if (!graph->getTensors().contains(input_id)) {
    throw error("No Tensor {} in processShapeNode");
  }
  auto input  = graph->getTensors().get(input_id);
  auto &shape = input->info.shape();

  data.resize(shape.size() * sizeof(int64_t));
  int64_t *int_data = reinterpret_cast<int64_t *>(data.data());

  for (int i = 0; i < shape.size(); i++) {
    int_data[i] = shape[i];
  }

  TensorInfo outInfo(DataType::INT64, {static_cast<int64_t>(shape.size())});
  graph->getTensors().addConstInit(node.output(0), outInfo, data.data());
}

void OnnxConstExprUtil::processConstantOfShapeNode(
    const ONNX_NAMESPACE::NodeProto &node,
    Graph *graph) {
  auto inputId     = node.input(0);
  auto inputTensor = graph->getTensors().get(inputId);
  TensorId name    = node.output(0);

  // Check the input tensor is of type INT64 and has data.
  if (inputTensor->info.dataType() != DataType::INT64) {
    throw error("expected data type {} but input tensor `{}` has data type {}.",
                DataType::INT64,
                inputId,
                inputTensor->info.dataType());
  }
  if (!inputTensor->hasTensorData()) {
    throw error("the tensor `" + inputId + "` does not have data");
  }
  TensorData *tensorData = inputTensor->tensorData();
  Shape outputShape =
      tensorData->copyDataAs<int64_t>(inputTensor->info.nelms());

  if (node.attribute().size() == 0) {
    // if no value provided, use DataType::FLOAT and value 0.0f
    TensorInfo resultInfo(DataType::FLOAT, outputShape);
    std::vector<float> resultData(resultInfo.nelms(), 0.0f);

    graph->getTensors().addConstInit(
        name, resultInfo, reinterpret_cast<void *>(resultData.data()));
  } else {
    // TensorData from attribute value
    const ONNX_NAMESPACE::TensorProto &value = node.attribute(0).t();
    ConstVoidData valueCVData                = onnxutil::getConstData(value);

    // Result takes data type from value and shape from input
    TensorInfo resultInfo(valueCVData.info.dataType(), outputShape);

    const char *valueData = reinterpret_cast<const char *>(valueCVData.data);
    std::vector<char> resultData(resultInfo.nbytes());
    for (int i = 0; i < resultData.size(); i++) {
      resultData.at(i) = valueData[i % valueCVData.info.nbytes()];
    }

    graph->getTensors().addConstInit(
        name, resultInfo, reinterpret_cast<void *>(resultData.data()));
  }
}

namespace {

static OpDefinition::DataTypes T = {
    DataType::UINT8,
    DataType::INT8,
    DataType::UINT16,
    DataType::INT16,
    DataType::INT32,
    DataType::INT64,
    DataType::UINT32,
    DataType::UINT64,
    DataType::BOOL,
    DataType::FLOAT,
    DataType::FLOAT16,
    DataType::BFLOAT16,
    DataType::DOUBLE,
    DataType::COMPLEX64,
    DataType::COMPLEX128,
    DataType::STRING,
};

static OpDefinition
    constantOpV9Def({OpDefinition::Inputs({}),
                     OpDefinition::Outputs({{"output", T}}),
                     OpDefinition::Attributes({{"value", {"*"}}})});

class ConstantOp {};

internal_error dummyOpError(const std::string &opName) {
  return internal_error("Can not create op of type '{}'. This op should be "
                        "handled by OnnxConstExprUtil",
                        opName);
}

static OpCreator<ConstantOp> constantOpCreator(
    OpDefinitions({{Onnx::Operators::Constant_9, constantOpV9Def}}),
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      throw dummyOpError("Constant");
    },
    true);

static OpDefinition
    shapeOpV1Def({OpDefinition::Inputs({{"data", T}}),
                  OpDefinition::Outputs({{"output", {DataType::INT64}}}),
                  OpDefinition::Attributes()});

class Shape {};

static OpCreator<Shape> shapeOpCreator(
    OpDefinitions({{Onnx::Operators::Shape_1, shapeOpV1Def}}),
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      throw dummyOpError("Shape");
    },
    true);

static OpDefinition constantOfShapeOpV9Def(
    {OpDefinition::Inputs({{"input", {DataType::INT64}}}),
     OpDefinition::Outputs({{"output", T}}),
     OpDefinition::Attributes({{"value", {"*"}}})});

class ConstantOfShape {};

static OpCreator<Shape> constantOfShape(
    OpDefinitions({{Onnx::Operators::ConstantOfShape_9,
                    constantOfShapeOpV9Def}}),
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      throw dummyOpError("ConstantOfShape");
    },
    true);

} // namespace

} // namespace popart
