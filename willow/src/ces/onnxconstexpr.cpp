// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <string>
#include <vector>
#include <popart/ces/onnxconstexpr.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

namespace popart {
class Op;

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
  TensorId name = addScope(*graph, node.output(0));
  // We assume that a tensor coming from a Constant Node should
  // not have a gradient computed for it or be updated during training
  // We will implement a separate tool to convert between
  // Constant Operator output and initializer in ONNX models (T6213)

  // Check for the `sparse_value` attribute.
  // `sparse_value` was introduces in opset 11 and is not currently supported.
  for (auto &attr : node.attribute()) {
    if (attr.name() == "sparse_value") {
      throw error("The Constant op attribute 'sparse_value' is not supported.");
    }
  }

  // Search the constant node for the 'value' attribute, and return the value
  // tensor.
  auto getValueAttribute = [](auto &nodeLocal) {
    for (auto &attr : nodeLocal.attribute()) {
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

  // Check the input tensor has data.
  if (!inputTensor->hasTensorData()) {
    throw error("the tensor `" + inputId + "` does not have data");
  }

  // If the data is int32 or int64, get the shape.
  // Otherwise error.
  TensorData *tensorData = inputTensor->tensorData();
  Shape outputShape;
  if (inputTensor->info.dataType() == DataType::INT64) {
    outputShape = tensorData->copyDataAs<int64_t>(
        static_cast<int>(inputTensor->info.nelms()));
  } else if (inputTensor->info.dataType() == DataType::INT32) {
    std::vector<int32_t> shapeData = tensorData->copyDataAs<int32_t>(
        static_cast<int>(inputTensor->info.nelms()));
    for (int32_t i : shapeData) {
      outputShape.push_back(i);
    }
  } else {
    throw error(
        "expected data type {} or {}, but input tensor `{}` has data type {}.",
        DataType::INT64,
        DataType::INT32,
        inputId,
        inputTensor->info.dataType());
  }

  // TODO : Create generic function to get attribute on certain name
  auto getValueAttribute = [](auto &nodeLocal) -> const onnx::TensorProto * {
    for (auto &attr : nodeLocal.attribute()) {
      if (attr.name() == "value") {
        return &attr.t();
      }
    }
    throw error("Could not find the 'value' attribute on the constant node");
  };

  const ONNX_NAMESPACE::TensorProto *value = getValueAttribute(node);

  if (value == nullptr) {
    // if no value provided, use DataType::FLOAT and value 0.0f
    TensorInfo resultInfo(DataType::FLOAT, outputShape);
    std::vector<float> resultData(resultInfo.nelms(), 0.0f);

    graph->getTensors().addConstInit(
        name, resultInfo, reinterpret_cast<void *>(resultData.data()));
  } else {
    // TensorData from attribute value
    ConstVoidData valueCVData = onnxutil::getConstData(*value);

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

static OpDefinition constantOpV11Def(
    {OpDefinition::Inputs({}),
     OpDefinition::Outputs({{"output", T}}),
     OpDefinition::Attributes({{"value", {"*"}}, {"value", {"*"}}})});

class ConstantOp {};

internal_error dummyOpError(const std::string &opName) {
  return internal_error("Can not create op of type '{}'. This op should be "
                        "handled by OnnxConstExprUtil",
                        opName);
}

static OpCreator<ConstantOp> constantOpCreator(
    OpDefinitions({{Onnx::Operators::Constant_9, constantOpV9Def},
                   {Onnx::Operators::Constant_11, constantOpV11Def}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
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
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
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
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      throw dummyOpError("ConstantOfShape");
    },
    true);

} // namespace

} // namespace popart
