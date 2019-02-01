#include <algorithm>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <poponnx/builder_impl.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

namespace poponnx {

// Supported IR version
const static uint64_t irVersion = 3;

// Generated opset versions
const static int64_t onnxOperatorSetVersion      = 9;
const static int64_t graphcoreOperatorSetVersion = 1;

// Supported opset versions
const static int64_t minOnnxOperatorSetVersion = 6;
const static int64_t maxOnnxOperatorSetVersion = 9;

const static int64_t minGraphcoreOperatorSetVersion = 1;
const static int64_t maxGraphcoreOperatorSetVersion = 1;

std::vector<std::string>
BuilderImpl::listConstExprNodesModel(const onnx::ModelProto &model,
                                     ExecutionMode mode) {
  const auto &graph = model.graph();

  std::unordered_set<std::string> initializers;
  for (const auto &init : graph.initializer()) {
    initializers.insert(init.name());
  }

  std::vector<std::string> nonConstExprIn;
  for (const auto &input : graph.input()) {
    if (initializers.count(input.name()) == 0) {
      nonConstExprIn.push_back(input.name());
    }
  }

  if (mode == ExecutionMode::TRAINING) {
    std::copy(initializers.begin(),
              initializers.end(),
              std::back_inserter(nonConstExprIn));
  }

  ConstExprUtil ce_util;
  auto constExprClassifier = ce_util.getClassifier(graph, nonConstExprIn);

  std::vector<std::string> result;
  for (const auto &node : graph.node()) {
    if (node.output_size() == 1 &&
        constExprClassifier.isConstExprTensor(node.output(0))) {
      result.push_back(node.output(0));
    }
  }

  return result;
}

std::vector<std::string>
BuilderImpl::listNonConstExprNodesModel(const onnx::ModelProto &model,
                                        ExecutionMode mode) {
  const auto const_nodes = listConstExprNodesModel(model, mode);
  std::unordered_set<std::string> const_nodes_ht;

  const_nodes_ht.reserve(const_nodes.size());
  const_nodes_ht.insert(const_nodes.begin(), const_nodes.end());

  std::vector<std::string> result;
  for (const auto &node : model.graph().node()) {
    if (const_nodes_ht.count(node.output(0)) == 0) {
      result.push_back(node.output(0));
    }
  }

  return result;
}

void BuilderImpl::finalizeOp(onnx::NodeProto *node, const std::string &name) {

  if (!name.empty())
    node->set_name(name);

  for (auto attribute : attributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  onnx::shape_inference::InferShapes(model_);
}

TensorId BuilderImpl::getNextId() {
  next_id_++;
  return std::to_string(next_id_);
}

void BuilderImpl::addOpsetRequirement(const std::string &domain, int version) {
  for (auto &o : model_.opset_import()) {
    if (o.domain() == domain && o.version() == version) {
      return;
    }
  }

  auto *opset = model_.add_opset_import();
  opset->set_domain(domain);
  opset->set_version(version);
}

BuilderImpl::BuilderImpl() {}

void BuilderImpl::configure() {
  next_id_ = 0;
  model_.set_ir_version(irVersion);

  addOpsetRequirement(ONNX_NAMESPACE::ONNX_DOMAIN, onnxOperatorSetVersion);

  model_.mutable_graph()->set_name("BuilderGraph");
}

TensorId BuilderImpl::addInputTensor(const TensorInfo &tensorInfo) {
  auto id             = getNextId();
  auto onnxTensorType = tensorInfo.getOnnxTypeProto();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = onnxTensorType;

  return id;
}

static void populateTensorProtoFromConstVoidData(const ConstVoidData &initData,
                                                 const std::string &id,
                                                 onnx::TensorProto *tp) {
  auto onnxTensorType = initData.info.getOnnxTypeProto();

  tp->set_data_type(onnxutil::getTPDataType(initData.info.dataType()));
  tp->set_name(id);

  for (auto d : initData.info.shape()) {
    tp->add_dims(d);
  }

  int element_count = static_cast<int>(initData.info.nelms());

  switch (initData.info.dataType()) {
  case DataType::FLOAT: {
    auto src = static_cast<const float *>(initData.data);
    auto dst = tp->mutable_float_data();
    dst->Resize(element_count, 0.0f);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::INT32: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = tp->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::INT64: {
    auto src = static_cast<const int64_t *>(initData.data);
    auto dst = tp->mutable_int64_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::BOOL: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = tp->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::FLOAT16: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = tp->mutable_int32_data();
    dst->Resize((element_count + 1) / 2, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::UNDEFINED:

  case DataType::STRING:
  case DataType::DOUBLE:
  case DataType::UINT32:
  case DataType::UINT64:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::BFLOAT16:
    throw error("Unsupported data type for initialized data ({})",
                initData.info);
  }
}

TensorId BuilderImpl::addInitializedInputTensor(const ConstVoidData &initData) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = initData.info.getOnnxTypeProto();

  auto *initializer = graph->add_initializer();
  populateTensorProtoFromConstVoidData(initData, id, initializer);

  return id;
}

void BuilderImpl::addOutputTensor(const TensorId &arg0) {
  auto *graph  = model_.mutable_graph();
  auto *output = graph->add_output();

  bool found = false;
  for (const auto &vi : graph->value_info()) {
    if (vi.name() == arg0) {
      *output = vi;
      found   = true;
    }
  }

  if (!found) {
    output->set_name(arg0);
  }
}

TensorId BuilderImpl::constant(const ConstVoidData &initData,
                               const std::string &name) {
  return op(Onnx::AiOnnx::OpSet9::Constant, {}, {{"value", initData}}, name)[0];
}

TensorId BuilderImpl::reshape_const(const std::vector<TensorId> &args,
                                    const std::vector<int64_t> &shape,
                                    const std::string &name) {
  Shape s = {static_cast<int64_t>(shape.size())};
  TensorInfo tensorInfo("INT64", s);
  auto newShape = constant({shape.data(), tensorInfo}, name + "_const");
  return op(Onnx::AiOnnx::OpSet9::Reshape, {args[0], newShape}, {}, name)[0];
}

std::vector<TensorId> BuilderImpl::customOp(
    const OperatorIdentifier &opid,
    const std::vector<boost::any> &inputs,
    const unsigned numOutputs,
    const std::vector<std::pair<std::string, boost::any>> &opAttributes,
    const std::string &name) {

  std::vector<TensorId> outputTensors(numOutputs);

  // Create the node
  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();

  // Set the domain/type
  node->set_op_type(opid.type);
  node->set_domain(opid.domain);

  // Add to the opset list
  addOpsetRequirement(opid.domain, opid.version);

  // Set the inputs
  for (auto input : inputs) {
    if (input.type() == typeid(TensorId)) {
      node->add_input(boost::any_cast<TensorId>(input));
    } else {
      throw error("Unknown input type {}", input.type().name());
    }
  }

  // Set the outputs
  for (int i = 0; i < numOutputs; ++i) {
    outputTensors[i] = getNextId();
    node->add_output(outputTensors[i]);
  }

  // Set the attributes
  for (auto attribute : opAttributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  finalizeOp(node, name);

  return outputTensors;
}

std::vector<TensorId> BuilderImpl::op(
    const OperatorIdentifier &opid,
    const std::vector<TensorId> &inputs,
    int numberOfOutputs,
    const std::map<std::string, boost::any> &opAttributes,
    const std::string &name,
    std::function<void(std::vector<TensorId>,
                       std::map<std::string, boost::any>)> validateInput) {

  if (opid.numInputs.min > 0) {
    // inputs.size  >= min inputs AND
    // if max != infinite then inputs <= max
    if (inputs.size() < opid.numInputs.min ||
        (opid.numInputs.max > 0 ? inputs.size() > opid.numInputs.max : false)) {
      throw error("{} has invalid number of inputs {}. Must be between {}..{}",
                  opid.type,
                  inputs.size(),
                  opid.numInputs.min,
                  ((opid.numInputs.max > 0) ? std::to_string(opid.numInputs.max)
                                            : "inf"));
    }
  }

  if (validateInput)
    validateInput(inputs, opAttributes);

  std::vector<TensorId> outputTensors(opid.numOutputs);

  // Create the node
  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();

  // Set the domain/type
  node->set_op_type(opid.type);

  // This is brittle, consider how to improve
  if (opid.domain != Domain::ai_onnx) {
    node->set_domain(opid.domain);
    addOpsetRequirement(Domain::ai_graphcore, 1);
  }

  // Set the inputs
  for (const auto &input : inputs) {
    node->add_input(input);
  }

  // Set the outputs
  for (int i = 0; i < numberOfOutputs; ++i) {
    outputTensors[i] = getNextId();
    node->add_output(outputTensors[i]);
  }

  // Set the attributes
  for (auto attribute : opAttributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  finalizeOp(node, name);

  return outputTensors;
}

bool BuilderImpl::findNodeProtoByOutputNamesImpl(
    onnx::NodeProto *&out,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::GraphProto *graph = model_.mutable_graph();
  for (onnx::NodeProto &node : *graph->mutable_node()) {
    // Don't check nodes which don't have the same number of outputs.
    if (node.output_size() != nodeOutputNames.size()) {
      continue;
    }

    // Match up all the outputs - note that output names are always unique so we
    // don't need to worry about the order.
    std::set<TensorId> unfoundNodeOutputNames = nodeOutputNames;
    for (const std::string &output : node.output()) {
      if (unfoundNodeOutputNames.count(output)) {
        unfoundNodeOutputNames.erase(output);
      }
    }

    // Return the node if we matched.
    if (unfoundNodeOutputNames.size() == 0) {
      out = &node;
      return true;
    }
  }
  return false;
}

onnx::NodeProto &BuilderImpl::findNodeProtoByOutputNames(
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto *node = nullptr;
  bool found            = findNodeProtoByOutputNamesImpl(node, nodeOutputNames);
  if (!found) {
    std::ostringstream stream;
    std::copy(nodeOutputNames.begin(),
              nodeOutputNames.end(),
              std::ostream_iterator<TensorId>(stream, " ,"));
    std::string s = stream.str();
    s.erase(s.length() - 2);
    throw error("Could not find a node with outputs " + s + ".");
  }
  return *node;
}

bool BuilderImpl::nodeHasAttributeImpl(onnx::AttributeProto *&out,
                                       onnx::NodeProto &node,
                                       const std::string &attributeName) {
  // Finds an attribute in a node.
  for (onnx::AttributeProto &attribute : *node.mutable_attribute()) {
    if (attribute.name().compare(attributeName) == 0) {
      out = &attribute;
      return true;
    }
  }
  return false;
}

bool BuilderImpl::nodeHasAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node      = findNodeProtoByOutputNames(nodeOutputNames);
  onnx::AttributeProto *attr = nullptr; // unused
  return nodeHasAttributeImpl(attr, node, attributeName);
}

onnx::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  return addNewAttributeToNode(attributeName, node);
}

onnx::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   onnx::NodeProto &node) {
  onnx::AttributeProto *attr = nullptr;
  bool hasAttribute          = nodeHasAttributeImpl(attr, node, attributeName);
  if (hasAttribute) {
    throw error("Node already has attribute " + attributeName + ".");
  }
  attr = node.add_attribute();
  attr->set_name(attributeName);
  return *attr;
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int64_t &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<int64_t> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INTS);
  for (int64_t i : attributeValue) {
    attr.add_ints(i);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const float &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::FLOAT);
  attr.set_f(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<float> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::FLOATS);
  for (float f : attributeValue) {
    attr.add_floats(f);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::string &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const char *attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(
    const std::string &attributeName,
    const std::vector<std::string> &attributeValue,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRINGS);
  for (std::string s : attributeValue) {
    attr.add_strings(s);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const bool attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(static_cast<int>(attributeValue));
}

// TODO change any to variant
void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const boost::any &attributeValue,
                                   onnx::NodeProto &node) {

  onnx::AttributeProto &attr = addNewAttributeToNode(attributeName, node);

  const std::type_info &tinfo = attributeValue.type();
  if (tinfo == typeid(int)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(boost::any_cast<int>(attributeValue));
  } else if (tinfo == typeid(int64_t)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(static_cast<int>(boost::any_cast<int64_t>(attributeValue)));
  } else if (tinfo == typeid(uint64_t)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(static_cast<int>(boost::any_cast<uint64_t>(attributeValue)));
  } else if (tinfo == typeid(std::vector<int64_t>)) {
    attr.set_type(onnx::AttributeProto::INTS);
    const std::vector<int64_t> &values =
        boost::any_cast<const std::vector<int64_t> &>(attributeValue);
    for (auto i : values) {
      attr.add_ints(i);
    }
  } else if (tinfo == typeid(float)) {
    attr.set_type(onnx::AttributeProto::FLOAT);
    attr.set_f(boost::any_cast<float>(attributeValue));
  } else if (tinfo == typeid(std::vector<float>)) {
    attr.set_type(onnx::AttributeProto::FLOAT);
    const std::vector<float> &values =
        boost::any_cast<const std::vector<float> &>(attributeValue);
    for (auto f : values) {
      attr.add_floats(f);
    }
  } else if (tinfo == typeid(std::string)) {
    attr.set_type(onnx::AttributeProto::STRING);
    attr.set_s(boost::any_cast<std::string>(attributeValue));
  } else if (tinfo == typeid(char *)) {
    attr.set_type(onnx::AttributeProto::STRING);
    attr.set_s(boost::any_cast<char *>(attributeValue));
  } else if (tinfo == typeid(std::vector<std::string>)) {
    attr.set_type(onnx::AttributeProto::STRINGS);
    const std::vector<std::string> &values =
        boost::any_cast<const std::vector<std::string> &>(attributeValue);
    for (auto &s : values) {
      attr.add_strings(s);
    }
  } else if (tinfo == typeid(bool)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(static_cast<int>(boost::any_cast<bool>(attributeValue)));
  } else if (tinfo == typeid(ConstVoidData)) {
    attr.set_type(onnx::AttributeProto::TENSOR);
    auto *t = attr.mutable_t();
    populateTensorProtoFromConstVoidData(
        boost::any_cast<ConstVoidData>(attributeValue), attributeName, t);
  } else {
    throw error("Unsupported attribute value type {}", tinfo.name());
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const ConstVoidData &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::TENSOR);

  auto *t = attr.mutable_t();
  populateTensorProtoFromConstVoidData(attributeValue, attributeName, t);
}

onnx::AttributeProto &
BuilderImpl::getNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node      = findNodeProtoByOutputNames(nodeOutputNames);
  onnx::AttributeProto *attr = nullptr;
  bool hasAttribute          = nodeHasAttributeImpl(attr, node, attributeName);
  if (!hasAttribute) {
    throw error("Node does not have an attribute " + attributeName + ".");
  }
  return *attr;
}

int64_t
BuilderImpl::getInt64NodeAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not an integer.");
  }
  return attr.i();
}

std::vector<int64_t> BuilderImpl::getInt64VectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<int64_t> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INTS) {
    throw error("Attribute " + attributeName + " is not an integer vector.");
  }
  for (int64_t i : attr.ints()) {
    out.push_back(i);
  }
  return out;
}

float BuilderImpl::getFloatNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::FLOAT) {
    throw error("Attribute " + attributeName + " is not a float.");
  }
  return attr.f();
}

std::vector<float> BuilderImpl::getFloatVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<float> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::FLOATS) {
    throw error("Attribute " + attributeName + " is not a float vector.");
  }
  for (float f : attr.floats()) {
    out.push_back(f);
  }
  return out;
}

std::string
BuilderImpl::getStringNodeAttribute(const std::string &attributeName,
                                    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::STRING) {
    throw error("Attribute " + attributeName + " is not a string.");
  }
  return attr.s();
}

std::vector<std::string> BuilderImpl::getStringVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<std::string> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::STRINGS) {
    throw error("Attribute " + attributeName + " is not a string vector.");
  }
  for (std::string s : attr.strings()) {
    out.push_back(s);
  }
  return out;
}

bool BuilderImpl::getBoolNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {

  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not a int.");
  }

  return static_cast<bool>(attr.i());
}

void BuilderImpl::removeNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  // To delete an attribute we must find the iterator for the attribute that we
  // want to delete.
  auto *attrs      = node.mutable_attribute();
  auto attr_it     = attrs->begin();
  auto attr_it_end = attrs->end();
  for (; attr_it != attr_it_end; attr_it++) {
    auto attr = *attr_it;
    if (attr.name().compare(attributeName) == 0) {
      break;
    }
  }
  if (attr_it != attr_it_end) {
    attrs->erase(attr_it);
  } else {
    throw error("Cannot remove attribute " + attributeName +
                " as it does not exist.");
  }
}

std::vector<std::string> BuilderImpl::getAllNodeAttributeNames(
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  std::vector<std::string> out;
  for (auto attr : node.attribute()) {
    out.push_back(attr.name());
  }
  return out;
}

void BuilderImpl::loadModelProto(const std::string &modelProtoOrFilename) {
  // TODO T5564 - merge the models rather than override the existing one.
  model_ = onnxutil::getModelProto(modelProtoOrFilename);

  // Check imported model is valid.
  onnx::checker::check_model(model_);

  // Check the IR version.
  if (model_.ir_version() != irVersion) {
    throw error("Expecting ONNX IR version {}, but got {}.",
                irVersion,
                model_.ir_version());
  }

  // Check the opset versions.
  for (auto opset : model_.opset_import()) {
    if (opset.domain() == "" && (opset.version() < minOnnxOperatorSetVersion ||
                                 opset.version() > maxOnnxOperatorSetVersion)) {
      throw error("Expecting ONNX opset version {}, but got {}.",
                  onnxOperatorSetVersion,
                  opset.version());
    }
    if (opset.domain() == Domain::ai_graphcore &&
        (opset.version() < minGraphcoreOperatorSetVersion ||
         opset.version() < maxGraphcoreOperatorSetVersion)) {
      throw error("Expecting GC opset version {}, but got {}.",
                  graphcoreOperatorSetVersion,
                  opset.version());
    }
  }
}

std::string BuilderImpl::getModelProto() const {
  std::string output;
  model_.SerializeToString(&output);
  return output;
}

std::vector<std::string>
BuilderImpl::listConstExprNodes(ExecutionMode mode) const {
  return listConstExprNodesModel(model_, mode);
}

std::vector<std::string>
BuilderImpl::listNonConstExprNodes(ExecutionMode mode) const {
  return listNonConstExprNodesModel(model_, mode);
}

std::vector<TensorId> BuilderImpl::getInputTensorIds() const {
  std::vector<TensorId> inNames;
  for (const auto &input : model_.graph().input()) {
    inNames.push_back(input.name());
  }
  return inNames;
}

std::vector<TensorId> BuilderImpl::getOutputTensorIds() const {
  std::vector<TensorId> outNames;
  for (const auto &output : model_.graph().output()) {
    outNames.push_back(output.name());
  }
  return outNames;
}

std::vector<TensorId> BuilderImpl::getValueTensorIds() const {
  std::vector<TensorId> valueNames;
  for (const auto &value_info : model_.graph().value_info()) {
    valueNames.push_back(value_info.name());
  }
  return valueNames;
}

bool BuilderImpl::isInputTensor(TensorId id) const {
  std::vector<TensorId> inIds = getInputTensorIds();
  return std::find(inIds.begin(), inIds.end(), id) != inIds.end();
}

bool BuilderImpl::isOutputTensor(TensorId id) const {
  std::vector<TensorId> outIds = getOutputTensorIds();
  return std::find(outIds.begin(), outIds.end(), id) != outIds.end();
}

bool BuilderImpl::isValueTensor(TensorId id) const {
  std::vector<TensorId> valueIds = getValueTensorIds();
  return std::find(valueIds.begin(), valueIds.end(), id) != valueIds.end();
}

std::string BuilderImpl::getStrFromTensorIdVec(std::vector<TensorId> v) const {
  const char *const delim = " ";
  std::ostringstream s;
  std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(s, delim));
  return s.str();
}

int BuilderImpl::getInputTensorIndex(TensorId id) const {
  if (isInputTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().input_size(); i++) {
      if (model_.graph().input(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an input tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getInputTensorIds()));
  }
}

int BuilderImpl::getOutputTensorIndex(TensorId id) const {
  if (isOutputTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().output_size(); i++) {
      if (model_.graph().output(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an output tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getOutputTensorIds()));
  }
}

int BuilderImpl::getValueTensorIndex(TensorId id) const {
  if (isValueTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().value_info_size(); i++) {
      if (model_.graph().value_info(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an value tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getOutputTensorIds()));
  }
}

const onnx::ValueInfoProto &BuilderImpl::getValueInfoProto(TensorId id) const {
  if (isInputTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().input(getInputTensorIndex(id));
    return t;
  } else if (isOutputTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().output(getOutputTensorIndex(id));
    return t;
  } else if (isValueTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().value_info(getValueTensorIndex(id));
    return t;
  } else {
    throw error("{} is not an known tensor. Must be one of {} {} {}",
                id,
                getStrFromTensorIdVec(getInputTensorIds()),
                getStrFromTensorIdVec(getOutputTensorIds()),
                getStrFromTensorIdVec(getValueTensorIds()));
  }
}

std::vector<int64_t> BuilderImpl::getTensorShape(const TensorId id) {
  std::vector<int64_t> shape;

  auto &t = getValueInfoProto(id);
  for (const auto &dim : t.type().tensor_type().shape().dim()) {
    shape.push_back(dim.dim_value());
  }
  return shape;
}

void BuilderImpl::setAttribute(const std::string &attribute, boost::any value) {
  attributes.insert(std::make_pair(attribute, value));
}

void BuilderImpl::clearAttribute(const std::string &attribute) {
  attributes.erase(attribute);
}

} // namespace poponnx
