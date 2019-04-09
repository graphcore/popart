#include <algorithm>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <poponnx/builder_impl.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

namespace poponnx {

// Supported IR version - this is the file format version
const static uint64_t minIrVersion = 3;
const static uint64_t maxIrVersion = 4;

// The Ir version will be set based on the version of the ai.onnx operator set.
// However if a graph is created without any operators from this domain the
// default Ir version will be as defined below.
const static uint64_t defaultIrVersion = 4;

// Default opset versions
const static int64_t onnxOperatorSetVersion      = defaultAiOnnxOpset;
const static int64_t graphcoreOperatorSetVersion = defaultAiGraphcoreOpset;

// Supported opset versions
const static int64_t minOnnxOperatorSetVersion = 6;
const static int64_t maxOnnxOperatorSetVersion = 9;

const static int64_t minGraphcoreOperatorSetVersion = 1;
const static int64_t maxGraphcoreOperatorSetVersion = 1;

void BuilderImpl::finalizeOp(onnx::NodeProto *node, const std::string &name) {

  std::string debug_name = name.empty() ? node->op_type() : name;

  std::stringstream fullname;
  for (const auto &n : name_scope_stack_) {
    fullname << n << ".";
  }
  fullname << debug_name;
  if (!name.empty() || !name_scope_stack_.empty()) {
    node->set_name(fullname.str());
  }

  for (auto attribute : attributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  onnx::shape_inference::InferShapes(model_);
}

std::map<BuilderImpl::NameStackIndex, int> BuilderImpl::tensorIdCounter = {};

void BuilderImpl::resetTensorIdCounter() { tensorIdCounter = {}; }

TensorId BuilderImpl::getNextId(const std::string &name, OutIndex n) {

  // obtain the stack state string
  std::stringstream stack_ss;
  for (const auto &s : name_scope_stack_) {
    stack_ss << s << '.';
  }
  std::string stack_str = stack_ss.str();

  // search for the count in the global map
  auto iter = tensorIdCounter.find(NameStackIndex(name, stack_str, n));

  // generate the unique id string
  std::stringstream id_ss;
  id_ss << stack_str << name << '_';
  if (iter == tensorIdCounter.end()) {
    id_ss << 0;
    tensorIdCounter[NameStackIndex(name, stack_str, n)] = 1;
  } else {
    id_ss << iter->second;
    ++(iter->second);
  }
  id_ss << ':' << std::to_string(n);
  std::string id = id_ss.str();
  return id;
}

void BuilderImpl::addOpsetRequirement(const std::string &domain, int version) {

  logging::builder::info(
      "Setting domain '{}' to opset version {}", domain, version);

  for (auto &o : model_.opset_import()) {
    if (o.domain() == domain && o.version() == version) {
      return;
    }
  }

  auto *opset = model_.add_opset_import();
  opset->set_domain(domain);
  opset->set_version(version);

  // Set the file format version based on the ai.onnx version
  // as per https://github.com/onnx/onnx/blob/master/docs/Versioning.md
  if (domain == Domain::ai_onnx) {
    if (version <= 8)
      model_.set_ir_version(3);
    else // if (version >= 9)
      model_.set_ir_version(4);
  }
}

void BuilderImpl::configure() {
  model_.set_ir_version(defaultIrVersion);

  model_.mutable_graph()->set_name("BuilderGraph");
}

TensorId BuilderImpl::addInputTensor(const TensorInfo &tensorInfo) {

  auto id = getNextId("input", 0);
  addInputTensorFromParentGraph(tensorInfo, id);
  return id;
}

void BuilderImpl::addInputTensorFromParentGraph(const TensorInfo &tensorInfo,
                                                const TensorId &tensorId) {

  auto onnxTensorType = tensorInfo.getOnnxTypeProto();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(tensorId);

  auto *type = input->mutable_type();
  *type      = onnxTensorType;
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
  case DataType::INT32:
  case DataType::UINT32: {
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
  case DataType::UINT64:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::BFLOAT16:
    throw error("Unsupported data type for initialized data ({})",
                initData.info);
  }
}

TensorId BuilderImpl::addInitializedInputTensor(const ConstVoidData &initData) {
  auto id = getNextId("init_input", 0);

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

std::vector<TensorId> BuilderImpl::op(
    const OperatorIdentifier &opid,
    int opsetVersion,
    const std::vector<TensorId> &inputs,
    const unsigned numberOfOutputs,
    const std::map<std::string, boost::any> &opAttributes,
    const std::string &name,
    std::function<void(std::vector<TensorId>,
                       std::map<std::string, boost::any>)> validateInput) {

  logging::builder::debug("Adding {} to builder opset:{}, numInputs:{} "
                          "numOutputs:{} numAttributes:{} name:{}",
                          opid,
                          opsetVersion,
                          inputs.size(),
                          numberOfOutputs,
                          opAttributes.size(),
                          name);

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

  std::vector<TensorId> outputTensors(numberOfOutputs);

  // Set the opset version for this domain if this is the first op in the domain
  // to be added else check that the opset is correct
  if (opsetVersions.find(opid.domain) == opsetVersions.end()) {
    // First op for this domain add the opset requirement
    addOpsetRequirement((opid.domain == Domain::ai_onnx ? "" : opid.domain),
                        opsetVersion);
    opsetVersions[opid.domain] = opsetVersion;
  } else {
    // For subsequent op's need to make sure it is the same version
    if (opsetVersions[opid.domain] != opsetVersion) {
      throw error("Invalid opset {} used to add an operation. Opset for domain "
                  "{} already defined as {}",
                  opsetVersion,
                  opid.domain,
                  opsetVersions[opid.domain]);
    }
  }

  // Create the node
  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();

  // Set the domain/type
  node->set_op_type(opid.type);

  // Set the domain if not ai.onnx
  if (opid.domain != Domain::ai_onnx) {
    node->set_domain(opid.domain);
  }

  // Set the inputs
  for (const auto &input : inputs) {
    node->add_input(input);
  }

  // Set the outputs
  for (int i = 0; i < numberOfOutputs; ++i) {
    outputTensors[i] = getNextId(opid.type, i);
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
    throw error("Could not find a node with outputs " + s +
                ". Must specify all outputs of a node");
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

  if (tinfo == typeid(int32_t)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(boost::any_cast<int32_t>(attributeValue));
  } else if (tinfo == typeid(uint32_t)) {
    attr.set_type(onnx::AttributeProto::INT);
    attr.set_i(boost::any_cast<uint32_t>(attributeValue));
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
  } else if (tinfo == typeid(onnx::GraphProto)) {
    attr.set_type(onnx::AttributeProto::GRAPH);
    const onnx::GraphProto &graph =
        boost::any_cast<const onnx::GraphProto &>(attributeValue);
    auto *g = attr.mutable_g();
    *g      = graph;
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
  if (model_.ir_version() < minIrVersion &&
      model_.ir_version() > maxIrVersion) {
    throw error("Expecting ONNX IR version {} to {}, but got {}.",
                minIrVersion,
                maxIrVersion,
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

    // TODO : Need to check if we have already set the domain opsetversion
    // Record which opset is defined in the model so that we can verify that we
    // add op's from the same opset
    opsetVersions[opset.domain() == "" ? Domain::ai_onnx : opset.domain()] =
        opset.version();
  }
}

std::string BuilderImpl::getModelProto() const {
  std::string output;
  model_.SerializeToString(&output);
  return output;
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

void BuilderImpl::pushNameScope(const std::string &name) {
  name_scope_stack_.push_back(name);
}

void BuilderImpl::popNameScope() { name_scope_stack_.pop_back(); }

} // namespace poponnx
