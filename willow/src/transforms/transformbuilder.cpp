// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <onnxutil.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/transforms/transformbuilder.hpp>

#include <popart/op/add.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/vendored/any.hpp>

namespace popart {

std::unique_ptr<Op>
TransformBuilder::createOp(const OperatorIdentifier &opid,
                           std::map<std::string, popart::any> attributes,
                           const std::string debugPrefix,
                           const std::vector<TensorId> &inIds) {
  return OpManager::createOpWithInputs(
      opid,
      graph,
      debugPrefix,
      OpManager::getAttributesFromAnyMap(attributes),
      inIds);
}

TensorId TransformBuilder::op(const OperatorIdentifier &_opid,
                              std::vector<TensorId> &inputs,
                              std::map<std::string, popart::any> attributes,
                              OptionalVGraphId virtualGraphId,
                              OptionalPipelineStage pipelineStage,
                              OptionalExecutionPhase executionPhase,
                              const std::string opName,
                              const std::string outputName) {

  auto op = createOp(_opid, attributes, opName, inputs);

  if (op == nullptr) {
    throw error("Failed to create op : {} in the transform builder", _opid);
  }

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->createAndConnectOutTensor(0, outputName);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }
  op->setPipelineStage(pipelineStage);
  op->setExecutionPhase(executionPhase);

  op->setup();
  auto _op = op.get();
  graph.moveIntoGraph(std::move(op));
  return _op->outTensor(0)->id;
}

void TransformBuilder::opWithOutput(
    const OperatorIdentifier &_opid,
    std::vector<TensorId> &inputs,
    std::map<std::string, popart::any> attributes,
    TensorId &out,
    OptionalVGraphId virtualGraphId,
    OptionalPipelineStage pipelineStage,
    OptionalExecutionPhase executionPhase,
    const std::string debugPrefix) {

  auto op = createOp(_opid, attributes, debugPrefix, inputs);

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->connectOutTensor(0, out);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }
  op->setPipelineStage(pipelineStage);
  op->setExecutionPhase(executionPhase);

  op->setup();
  graph.moveIntoGraph(std::move(op));
}

std::vector<TensorId>
TransformBuilder::multiOutputOp(const OperatorIdentifier &_opid,
                                std::vector<TensorId> &inputs,
                                OutIndex numberOfOutputs,
                                std::map<std::string, popart::any> attributes,
                                OptionalVGraphId virtualGraphId,
                                OptionalPipelineStage pipelineStage,
                                OptionalExecutionPhase executionPhase,
                                const std::string opName) {

  auto op = createOp(_opid, attributes, opName, inputs);

  if (op == nullptr) {
    throw error("Failed to create op : {} in the transform builder", _opid);
  }

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  std::vector<TensorId> result;
  for (OutIndex i = 0; i < numberOfOutputs; i++) {
    op->createAndConnectOutTensor(
        i, op->getIr().createIntermediateTensorId(inputs.at(0)));
    result.push_back(op->outId(i));
  }

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }
  op->setPipelineStage(pipelineStage);
  op->setExecutionPhase(executionPhase);

  op->setup();
  graph.moveIntoGraph(std::move(op));

  return result;
}

TransformBuilder::TransformBuilder(Graph &graph_) : graph(graph_) {}

TensorId TransformBuilder::getNextId(const std::string &name, OutIndex n) {
  std::stringstream id_ss;
  id_ss << name;

  if (n >= 0) {
    id_ss << ':' << std::to_string(n);
  }

  bool valid       = false;
  auto proposedStr = id_ss.str();
  int c            = 0;
  while (!valid) {

    if (c != 0) {
      proposedStr = id_ss.str() + "/" + std::to_string(c);
    }

    valid = (graph.getIr().containsTensor(proposedStr) == false);

    c++;
  }

  return proposedStr;
}

TensorId TransformBuilder::concat(std::vector<TensorId> &inputs,
                                  int64_t axis,
                                  OptionalVGraphId virtualGraphId,
                                  OptionalPipelineStage pipelineStage,
                                  OptionalExecutionPhase executionPhase,
                                  const std::string opName,
                                  const std::string outputName) {
  return op(Onnx::Operators::Concat_1,
            inputs,
            {{"axis", static_cast<int64_t>(axis)}},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

void TransformBuilder::concat(std::vector<TensorId> &inputs,
                              int64_t axis,
                              TensorId out,
                              OptionalVGraphId virtualGraphId,
                              OptionalPipelineStage pipelineStage,
                              OptionalExecutionPhase executionPhase,
                              const std::string opName) {
  opWithOutput(Onnx::Operators::Concat_1,
               inputs,
               {{"axis", static_cast<int64_t>(axis)}},
               out,
               virtualGraphId,
               pipelineStage,
               executionPhase,
               opName);
}

void TransformBuilder::sum(std::vector<TensorId> &inputs,
                           TensorId out,
                           OptionalVGraphId virtualGraphId,
                           OptionalPipelineStage pipelineStage,
                           OptionalExecutionPhase executionPhase,
                           const std::string opName) {
  opWithOutput(Onnx::Operators::Sum_8,
               inputs,
               {},
               out,
               virtualGraphId,
               pipelineStage,
               executionPhase,
               opName);
}

TensorId TransformBuilder::addLhsInplace(std::vector<TensorId> &inputs,
                                         OptionalVGraphId virtualGraphId,
                                         OptionalPipelineStage pipelineStage,
                                         OptionalExecutionPhase executionPhase,
                                         const std::string opName,
                                         const std::string outputName) {
  Op::Settings settings(graph, opName);

  auto op = std::make_unique<AddLhsInplaceOp>(settings);

  if (op == nullptr) {
    throw error(
        "Failed to create op : AddLhsInplaceOp in the transform builder");
  }

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->createAndConnectOutTensor(0, outputName);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }

  if (pipelineStage) {
    op->setPipelineStage(*pipelineStage);
  }

  if (executionPhase) {
    op->setExecutionPhase(*executionPhase);
  }

  op->setup();
  auto _op = op.get();
  graph.moveIntoGraph(std::move(op));
  return _op->outTensor(0)->id;
}

void TransformBuilder::addLhsInplace(std::vector<TensorId> &inputs,
                                     TensorId out,
                                     OptionalVGraphId virtualGraphId,
                                     OptionalPipelineStage pipelineStage,
                                     OptionalExecutionPhase executionPhase,
                                     const std::string opName) {
  Op::Settings settings(graph, opName);

  auto op = std::make_unique<AddLhsInplaceOp>(settings);

  if (op == nullptr) {
    throw error(
        "Failed to create op : AddLhsInplaceOp in the transform builder");
  }

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->connectOutTensor(0, out);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }

  if (pipelineStage) {
    op->setPipelineStage(*pipelineStage);
  }

  if (executionPhase) {
    op->setExecutionPhase(*executionPhase);
  }

  op->setup();
  graph.moveIntoGraph(std::move(op));
}

TensorId TransformBuilder::matmul(TensorId lhs,
                                  TensorId rhs,
                                  OptionalVGraphId virtualGraphId,
                                  OptionalPipelineStage pipelineStage,
                                  OptionalExecutionPhase executionPhase,
                                  const std::string opName,
                                  const std::string outputName,
                                  std::map<std::string, popart::any> attrs,
                                  const OperatorIdentifier _opid) {
  std::vector<TensorId> inputs = {lhs, rhs};
  return op(_opid,
            inputs,
            attrs,
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

void TransformBuilder::cast(TensorId input,
                            TensorId out,
                            DataType type,
                            OptionalVGraphId virtualGraphId,
                            OptionalPipelineStage pipelineStage,
                            OptionalExecutionPhase executionPhase,
                            const std::string opName) {
  std::vector<TensorId> inputs = {input};
  opWithOutput(
      Onnx::Operators::Cast_9,
      inputs,
      {{"to", static_cast<Attributes::Int>(onnxutil::getTPDataType(type))}},
      out,
      virtualGraphId,
      pipelineStage,
      executionPhase,
      opName);
}

TensorId TransformBuilder::slice(TensorId in,
                                 const Shape &starts,
                                 const Shape &ends,
                                 const Shape &axes,
                                 OptionalVGraphId virtualGraphId,
                                 OptionalPipelineStage pipelineStage,
                                 OptionalExecutionPhase executionPhase,
                                 const std::string opName,
                                 const std::string outputName) {
  std::vector<TensorId> inputs = {in};

  return op(Onnx::Operators::Slice_1,
            inputs,
            {{"starts", starts}, {"ends", ends}, {"axes", axes}},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

TensorId TransformBuilder::sliceInPlace(TensorId in,
                                        const Shape &starts,
                                        const Shape &ends,
                                        const Shape &axes,
                                        OptionalVGraphId virtualGraphId,
                                        OptionalPipelineStage pipelineStage,
                                        OptionalExecutionPhase executionPhase,
                                        const std::string opName,
                                        const std::string outputName) {
  std::vector<TensorId> inputs = {in};

  Op::Settings settings(graph, opName);

  auto op =
      std::make_unique<SliceInplaceOp>(Onnx::CustomOperators::SliceInplace,
                                       std::vector<int64_t>{starts}, // starts
                                       std::vector<int64_t>{ends},   // ends
                                       std::vector<int64_t>{axes},   // axes
                                       std::vector<int64_t>{},       // flips
                                       settings);

  if (op == nullptr) {
    throw error("Failed to create op : {} in the transform builder",
                Onnx::CustomOperators::SliceInplace);
  }

  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->createAndConnectOutTensor(0, outputName);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }

  if (pipelineStage) {
    op->setPipelineStage(*pipelineStage);
  }

  if (executionPhase) {
    op->setExecutionPhase(*executionPhase);
  }

  op->setup();
  auto _op = op.get();
  graph.moveIntoGraph(std::move(op));
  return _op->outTensor(0)->id;
}

void TransformBuilder::slice(TensorId in,
                             const Shape &starts,
                             const Shape &ends,
                             const Shape &axes,
                             TensorId out,
                             OptionalVGraphId virtualGraphId,
                             OptionalPipelineStage pipelineStage,
                             OptionalExecutionPhase executionPhase,
                             const std::string opName) {
  std::vector<TensorId> inputs = {in};

  opWithOutput(Onnx::Operators::Slice_1,
               inputs,
               {{"starts", starts}, {"ends", ends}, {"axes", axes}},
               out,
               virtualGraphId,
               pipelineStage,
               executionPhase,
               opName);
}

TensorId TransformBuilder::squeeze(TensorId in,
                                   const Shape &axes,
                                   OptionalVGraphId virtualGraphId,
                                   OptionalPipelineStage pipelineStage,
                                   OptionalExecutionPhase executionPhase,
                                   const std::string opName,
                                   const std::string outputName) {
  std::vector<TensorId> inputs = {in};

  return op(Onnx::Operators::Squeeze_1,
            inputs,
            {{"axes", axes}},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

void TransformBuilder::squeeze(TensorId in,
                               const Shape &axes,
                               TensorId out,
                               OptionalVGraphId virtualGraphId,
                               OptionalPipelineStage pipelineStage,
                               OptionalExecutionPhase executionPhase,
                               const std::string opName) {
  std::vector<TensorId> inputs = {in};

  opWithOutput(Onnx::Operators::Squeeze_1,
               inputs,
               {{"axes", axes}},
               out,
               virtualGraphId,
               pipelineStage,
               executionPhase,
               opName);
}

TensorId TransformBuilder::transpose(TensorId in,
                                     Shape perm,
                                     OptionalVGraphId virtualGraphId,
                                     OptionalPipelineStage pipelineStage,
                                     OptionalExecutionPhase executionPhase,
                                     const std::string opName,
                                     const std::string outTensorName) {
  std::vector<TensorId> inputs = {in};
  return op(Onnx::Operators::Transpose_1,
            inputs,
            {{"perm", perm}},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outTensorName);
}

void TransformBuilder::transpose(TensorId in,
                                 Shape perm,
                                 TensorId out,
                                 OptionalVGraphId virtualGraphId,
                                 OptionalPipelineStage pipelineStage,
                                 OptionalExecutionPhase executionPhase,
                                 const std::string opName) {
  std::vector<TensorId> inputs = {in};
  opWithOutput(Onnx::Operators::Transpose_1,
               inputs,
               {{"perm", perm}},
               out,
               virtualGraphId,
               pipelineStage,
               executionPhase,
               opName);
}

TensorId TransformBuilder::reshape(TensorId in,
                                   Shape shape,
                                   OptionalVGraphId virtualGraphId,
                                   OptionalPipelineStage pipelineStage,
                                   OptionalExecutionPhase executionPhase,
                                   const std::string opName,
                                   const std::string outputName) {
  auto op = createOp(Onnx::Operators::Reshape_5, {}, opName, {in});

  if (op == nullptr) {
    throw error("Failed to create op : {}", Onnx::Operators::Reshape_5);
  }

  ReshapeOp *reshape = dynamic_cast<ReshapeOp *>(op.get());

  // Have to duplicat this code so the setOutShape can be called
  reshape->setOutShape(shape);

  std::vector<TensorId> inputs = {in};
  for (int i = 0; i < inputs.size(); ++i) {
    op->connectInTensor(i, inputs[i]);
  }
  op->createAndConnectOutTensor(0, outputName);

  if (virtualGraphId) {
    op->setVirtualGraphId(*virtualGraphId);
  }
  op->setPipelineStage(pipelineStage);
  op->setExecutionPhase(executionPhase);

  op->setup();
  auto _op = op.get();
  graph.moveIntoGraph(std::move(op));
  return _op->outTensor(0)->id;
}

TensorId TransformBuilder::reducesum(TensorId in,
                                     int64_t keepdims,
                                     std::vector<int64_t> axes,
                                     OptionalVGraphId virtualGraphId,
                                     OptionalPipelineStage pipelineStage,
                                     OptionalExecutionPhase executionPhase,
                                     const std::string opName,
                                     const std::string outputName) {
  std::vector<TensorId> inputs = {in};
  return op(Onnx::Operators::ReduceSum_1,
            inputs,
            {{"keepdims", keepdims}, {"axes", axes}},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

std::vector<TensorId>
TransformBuilder::split(TensorId in,
                        int64_t axis,
                        std::vector<int64_t> splitSizes,
                        OptionalVGraphId virtualGraphId,
                        OptionalPipelineStage pipelineStage,
                        OptionalExecutionPhase executionPhase,
                        const std::string opName) {

  std::vector<TensorId> inputs = {in};
  return multiOutputOp(Onnx::Operators::Split_11,
                       inputs,
                       static_cast<int>(splitSizes.size()),
                       {{"axis", axis}, {"split", splitSizes}},
                       virtualGraphId,
                       pipelineStage,
                       executionPhase,
                       opName);
}

TensorId TransformBuilder::add(std::vector<TensorId> &inputs,
                               OptionalVGraphId virtualGraphId,
                               OptionalPipelineStage pipelineStage,
                               OptionalExecutionPhase executionPhase,
                               const std::string opName,
                               const std::string outputName) {
  return op(Onnx::Operators::Add_7,
            inputs,
            {},
            virtualGraphId,
            pipelineStage,
            executionPhase,
            opName,
            outputName);
}

void TransformBuilder::unsqueeze(TensorId in,
                                 std::vector<int64_t> axes,
                                 TensorId out,
                                 OptionalVGraphId virtualGraphId,
                                 OptionalPipelineStage pipelineStage,
                                 OptionalExecutionPhase executionPhase,
                                 const std::string opName) {
  std::vector<TensorId> inputs = {in};
  return opWithOutput(Onnx::Operators::Unsqueeze_11,
                      inputs,
                      {{"axes", axes}},
                      out,
                      virtualGraphId,
                      pipelineStage,
                      executionPhase,
                      opName);
}

Op *TransformBuilder::getProducer(TensorId id) {
  return graph.getIr().getTensor(id)->getProducer();
}

bool TransformBuilder::hasProducer(TensorId id) {
  return graph.getIr().getTensor(id)->hasProducer();
}

} // namespace popart
