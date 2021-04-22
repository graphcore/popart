// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <popart/builder_impl.hpp>
#include <popart/builderdebuginfo.hpp>
#include <popart/ces/constexpr.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/onnxutil.hpp>
#include <popart/opidentifier.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>

#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

namespace {
std::string generateUniqueGraphName(const std::string &name) {
  static int uid = 0;
  return name + "_" + std::to_string(uid++);
}
} // namespace

namespace popart {

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
const static int64_t maxOnnxOperatorSetVersion = 11;

const static int64_t minGraphcoreOperatorSetVersion = 1;
const static int64_t maxGraphcoreOperatorSetVersion = 1;

const BuilderImpl *BuilderImpl::getParent() const {
  if (!hasParent()) {
    throw internal_error("No Parent Builder");
  }
  return parent;
}

std::vector<const BuilderImpl *> BuilderImpl::getChildren() const {
  return children;
}

void BuilderImpl::finalizeOp(ONNX_NAMESPACE::NodeProto *node,
                             const OperatorIdentifier &opid,
                             const std::string &name) {

  std::string debug_name = name.empty() ? node->op_type() : name;

  std::stringstream fullname;
  for (const auto &n : name_scope_stack_) {
    fullname << n << sNameDelimiter;
  }

  fullname << debug_name;

  if (!name.empty() || !name_scope_stack_.empty()) {
    node->set_name(fullname.str());
  }

  for (auto attribute : attributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  // The node outputs are added to the model's value_info field here
  runShapeInference(node, opid);

  // Sanity check: verify the output dimensions of each output are valid
  for (int i = 0; i < node->output_size(); ++i) {
    const TensorId &output = node->output(i);
    // TODO T17932 : We do not have a mechanism for inferring the output shape
    // of custom ops, so this check can only be applied if the tensor shape
    // is known
    if (hasTensorShape(output)) {
      std::vector<int64_t> shape = getTensorShape(output);
      if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) {
            return dim < 0;
          })) {
        throw error(
            "Output '{}' of node '{}' has invalid shape, {}. Values must "
            "be non-negative in each shape dimension.",
            output,
            fullname.str(),
            shape);
      }
    }
  }
}

void BuilderImpl::runShapeInference(ONNX_NAMESPACE::NodeProto *node,
                                    const OperatorIdentifier &opid) {
  if (ShapeInferenceFunctions::hasFunction(opid)) {
    auto func = ShapeInferenceFunctions::getFunction(opid);
    // Create and prepare a ShapeInferenceContext.
    std::map<int, TensorInfo> inputInfos;
    for (int i = 0; i < node->input_size(); i++) {
      auto id    = node->input(i);
      auto shape = getTensorShape(id);
      auto dtype = getTensorDataType(id);
      inputInfos.insert({i, {dtype, shape}});
    }
    ShapeInferenceContext ctx(
        inputInfos, node->attribute(), node->output_size());
    // Call the inference function.
    func(ctx);
    // Create the ValueInfoProtos for the nodes outputs.
    for (auto &idx_info : ctx.getOutputInfos()) {
      auto idx   = idx_info.first;
      auto &info = idx_info.second;

      // Create the ValueInfoProto and set the name.
      auto valueInfo = model_.mutable_graph()->add_value_info();
      valueInfo->set_name(node->output(idx));
      // Create and get the TensorTypeProto.
      auto tt = valueInfo->mutable_type()->mutable_tensor_type();
      // Set the shape (this assumes it is clear to start with).
      auto shape = tt->mutable_shape();
      for (auto value : info.shape()) {
        shape->add_dim()->set_dim_value(value);
      }
      // Set the data type.
      tt->set_elem_type(onnxutil::getTPDataType(info.dataType()));
    }
  } else {
    ONNX_NAMESPACE::shape_inference::InferShapes(model_);
  }

  auto getValueInfo =
      [&](const TensorId &id) -> const ONNX_NAMESPACE::ValueInfoProto * {
    for (int i = 0; i < model_.graph().value_info_size(); i++) {
      if (model_.graph().value_info(i).name() == id) {
        return &model_.graph().value_info(i);
      }
    }
    return nullptr;
  };

  // Check shape inference worked.
  for (int i = 0; i < node->output_size(); i++) {
    auto &id         = node->output(i);
    auto *value_info = getValueInfo(id);
    if (!value_info) {
      logging::builder::warn(
          "Shape inference failed for output '{}' of node '{}'",
          id,
          node->op_type());
    }
  }
}

bool BuilderImpl::inHigherScope(const TensorId &id) const {

  if (hasParent()) {
    return getParent()->inCurrentScope(id) || getParent()->inHigherScope(id);
  }
  return false;
}

bool BuilderImpl::inLowerScope(const TensorId &id) const {

  for (auto &child : getChildren()) {
    if (child->inCurrentScope(id) || child->inLowerScope(id)) {
      return true;
    }
  }

  return false;
}

bool BuilderImpl::inCurrentScope(const TensorId &id) const {

  if (tensorIds.count(id) == 0) {
    return false;
  }
  return true;
}

TensorId BuilderImpl::getNextId(const std::string &name, OutIndex n) {

  // obtain the stack state string
  std::stringstream stack_ss;
  for (const auto &s : name_scope_stack_) {
    stack_ss << s << sNameDelimiter;
  }

  std::string stack_str = stack_ss.str();

  // generate the unique id string
  std::stringstream id_ss;
  id_ss << stack_str << name;

  // Add ':n' if the index n is >= 0
  if (n >= 0) {
    id_ss << ':' << std::to_string(n);
  }

  int counter = -1;
  bool valid  = false;

  std::string baseId  = id_ss.str();
  TensorId proposedId = baseId;
  while (!valid) {
    ++counter;
    if (counter != 0) {
      proposedId = baseId + sNameDelimiter + std::to_string(counter);
    }
    // Design decision: follow rules of block scoping of variables in C++,
    // disallowing overshadowing. This onnx spec is more relaxed, and only
    // requires unique tensor names within a scope. However, this should
    // improve the readability of the IR
    valid = (!inCurrentScope(proposedId) && !inLowerScope(proposedId) &&
             !inHigherScope(proposedId));
  }
  if (tensorIds.count(proposedId) != 0) {
    throw error("cannot re-use proposedId");
  }
  tensorIds.emplace(proposedId);
  return proposedId;
}

TensorId BuilderImpl::getNextInputId(const std::string &debugPrefix) {
  // Should we check for uniqueness of name? TODO T8278
  std::string name = debugPrefix.empty() ? "input" : debugPrefix;
  return getNextId(name);
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
  model_.mutable_graph()->set_name(generateUniqueGraphName("BuilderGraph"));
}

void BuilderImpl::setGraphName(const std::string &name) {
  std::string new_name = name.empty() ? generateUniqueGraphName(name) : name;
  model_.mutable_graph()->set_name(new_name);
}

TensorId BuilderImpl::addInputTensor(const TensorInfo &tensorInfo,
                                     const popart::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  auto id                = getNextInputId(debugPrefix);

  // note that a TypeProto contains both shape and numerical type
  ONNX_NAMESPACE::TypeProto onnxTensorType = tensorInfo.getOnnxTypeProto();

  // set name
  auto *input = addGraphInput(id);

  // set type
  auto *type = input->mutable_type();
  *type      = onnxTensorType;

  // Add the debug_id as meta data
  popart::BuilderVarDebugInfo di(
      debugContext, "addInputTensor", id, tensorInfo);

  auto meta_data = model_.mutable_metadata_props();
  auto a         = meta_data->Add();
  a->set_key(std::string(onnxDebugIdInputMetaDataKey) + id);
  a->set_value(std::to_string(di.getId()));

  return id;
}

TensorId
BuilderImpl::addUntypedInputTensor(const popart::DebugContext &debugContext) {
  // TODO : Check T8276
  // In the onnx spec:
  //     message ValueInfoProto {
  //     ...
  //       // This field MUST be present in this version of the IR for
  //       // inputs and outputs of the top-level graph.
  //       TypeProto type = 2;
  //
  // It looks like it should be fine to add an untyped input tensor on
  // subgraphs, but not in the top-level graph.
  if (!hasParent()) {
    throw error("Can not add untyped tensors to the top-level graph. Use "
                "Builder::addInputTensor(const TensorInfo &tensorInfo, const "
                "std::string &debugPrefix) instead.");
  }
  const auto debugPrefix = debugContext.getPathName();
  auto id                = getNextInputId(debugPrefix);
  addGraphInput(id);
  return id;
}

void BuilderImpl::addInputTensorFromParentGraph(const TensorId &tensorId) {

  // Should we check for uniqueness of name? TODO T8278

  if (!inHigherScope(tensorId)) {
    throw error(
        "Failed to add unrecognised Tensor {} from higher scope, "
        "Currently, a Tensor must already exist in a higher (parent) scope "
        "to add it as an input in a lower scope.",
        tensorId);
  }

  addGraphInput(tensorId);

  // set type
  // We need to run type inference to determine the DataType
  // According to the spec the type (input->mutable_type()) is NOT optional
  // TODO : get the type. T8276
}

ONNX_NAMESPACE::ValueInfoProto *BuilderImpl::addGraphInput(const TensorId &id) {
  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  return input;
}

void BuilderImpl::populateTensorProtoFromConstVoidData(
    const ConstVoidData &initData,
    const std::string &id,
    ONNX_NAMESPACE::TensorProto *tp) {
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
  case DataType::UINT32:
  case DataType::UINT64: {
    auto src = static_cast<const uint64_t *>(initData.data);
    auto dst = tp->mutable_uint64_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::UNDEFINED:
  case DataType::STRING:
  case DataType::DOUBLE:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::BFLOAT16:
    throw error("Unsupported data type for initialized data ({})",
                initData.info);
  }
}

TensorId BuilderImpl::addInitializedInputTensor(
    const ConstVoidData &initData,
    const popart::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  std::string name       = debugPrefix.empty() ? "init_input" : debugPrefix;

  auto id = getNextId(name);

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = initData.info.getOnnxTypeProto();

  auto *initializer = graph->add_initializer();
  populateTensorProtoFromConstVoidData(initData, id, initializer);

  // Add the debug_id as meta data
  popart::BuilderVarDebugInfo di(debugContext, "addInitializedInputTensor", id);

  auto meta_data = model_.mutable_metadata_props();
  auto a         = meta_data->Add();
  a->set_key(std::string(onnxDebugIdInputMetaDataKey) + id);
  a->set_value(std::to_string(di.getId()));

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

void BuilderImpl::op(
    const OperatorIdentifier &opid,
    int opsetVersion,
    const std::vector<TensorId> &inputs,
    const std::vector<TensorId> &outputs,
    const std::map<std::string, popart::any> &opAttributes,
    const DebugContext &debugContext,
    std::function<void(std::vector<TensorId>,
                       std::map<std::string, popart::any>)> validateInput) {

  auto name = debugContext.getPathName();
  logging::builder::debug("Adding {} to builder opset:{}, numInputs:{} "
                          "numOutputs:{} numAttributes:{} name:{}",
                          opid,
                          opsetVersion,
                          inputs.size(),
                          outputs.size(),
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

  // Get valid tensors of mutable graph.
  auto validTensors = getValidInputTensorIds();

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input     = inputs[i];
    auto isOptional = (i >= opid.numInputs.min);

    if (isOptional && input == "") {
      // Assume "" is used to indicate tensor is not set.
    } else {
      // Throw an exception if an tensor id is given as an input but it does not
      // exist.
      if (validTensors.find(input) == validTensors.end()) {
        std::stringstream ss;
        ss << "Unknown tensor '" << input << "' (";

        if (validTensors.size() == 0) {
          ss << "there are no valid tensors yet";
        } else {
          ss << "valid tensors are ";

          bool isFirst = true;
          for (auto id : validTensors) {
            if (!isFirst) {
              ss << ", ";
            } else {
              isFirst = false;
            }
            ss << id;
          }
        }

        ss << ").";
        throw error(ss.str());
      }
    }
  }

  if (validateInput)
    validateInput(inputs, opAttributes);

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
  for (auto &output : outputs) {
    node->add_output(output);
  }

  // Set the attributes
  for (auto attribute : opAttributes) {
    addNodeAttribute(attribute.first, attribute.second, *node);
  }

  finalizeOp(node, opid, name);
}

bool BuilderImpl::findNodeProtoByOutputNamesImpl(
    ONNX_NAMESPACE::NodeProto *&out,
    const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::GraphProto *graph = model_.mutable_graph();
  for (ONNX_NAMESPACE::NodeProto &node : *graph->mutable_node()) {
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

ONNX_NAMESPACE::NodeProto &BuilderImpl::findNodeProtoByOutputNames(
    const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::NodeProto *node = nullptr;
  bool found = findNodeProtoByOutputNamesImpl(node, nodeOutputNames);
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

bool BuilderImpl::nodeHasAttributeImpl(ONNX_NAMESPACE::AttributeProto *&out,
                                       ONNX_NAMESPACE::NodeProto &node,
                                       const std::string &attributeName) {
  // Finds an attribute in a node.
  for (ONNX_NAMESPACE::AttributeProto &attribute : *node.mutable_attribute()) {
    if (attribute.name().compare(attributeName) == 0) {
      out = &attribute;
      return true;
    }
  }
  return false;
}

bool BuilderImpl::nodeHasAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  ONNX_NAMESPACE::AttributeProto *attr = nullptr; // unused
  return nodeHasAttributeImpl(attr, node, attributeName);
}

ONNX_NAMESPACE::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  return addNewAttributeToNode(attributeName, node);
}

ONNX_NAMESPACE::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   ONNX_NAMESPACE::NodeProto &node) {
  ONNX_NAMESPACE::AttributeProto *attr = nullptr;
  bool hasAttribute = nodeHasAttributeImpl(attr, node, attributeName);
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
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<int64_t> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  for (int64_t i : attributeValue) {
    attr.add_ints(i);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const float &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
  attr.set_f(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<float> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOATS);
  for (float f : attributeValue) {
    attr.add_floats(f);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::string &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const char *attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(
    const std::string &attributeName,
    const std::vector<std::string> &attributeValue,
    const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);
  for (std::string s : attributeValue) {
    attr.add_strings(s);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const bool attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  attr.set_i(static_cast<int>(attributeValue));
}

// TODO change any to variant
void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const popart::any &attributeValue,
                                   ONNX_NAMESPACE::NodeProto &node) {

  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, node);

  const std::type_info &tinfo = attributeValue.type();

  if (tinfo == typeid(int32_t)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr.set_i(popart::any_cast<int32_t>(attributeValue));
  } else if (tinfo == typeid(uint32_t)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr.set_i(popart::any_cast<uint32_t>(attributeValue));
  } else if (tinfo == typeid(int64_t)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr.set_i(static_cast<int>(popart::any_cast<int64_t>(attributeValue)));
  } else if (tinfo == typeid(uint64_t)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr.set_i(static_cast<int>(popart::any_cast<uint64_t>(attributeValue)));
  } else if (tinfo == typeid(std::vector<int64_t>)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INTS);
    const std::vector<int64_t> &values =
        popart::any_cast<const std::vector<int64_t> &>(attributeValue);
    for (auto i : values) {
      attr.add_ints(i);
    }
  } else if (tinfo == typeid(float)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    attr.set_f(popart::any_cast<float>(attributeValue));
  } else if (tinfo == typeid(std::vector<float>)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::FLOATS);
    const std::vector<float> &values =
        popart::any_cast<const std::vector<float> &>(attributeValue);
    for (auto f : values) {
      attr.add_floats(f);
    }
  } else if (tinfo == typeid(std::string)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr.set_s(popart::any_cast<std::string>(attributeValue));
  } else if (tinfo == typeid(char *)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attr.set_s(popart::any_cast<char *>(attributeValue));
  } else if (tinfo == typeid(std::vector<std::string>)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);
    const std::vector<std::string> &values =
        popart::any_cast<const std::vector<std::string> &>(attributeValue);
    for (auto &s : values) {
      attr.add_strings(s);
    }
  } else if (tinfo == typeid(bool)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attr.set_i(static_cast<int>(popart::any_cast<bool>(attributeValue)));
  } else if (tinfo == typeid(ConstVoidData)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
    auto *t = attr.mutable_t();
    populateTensorProtoFromConstVoidData(
        popart::any_cast<ConstVoidData>(attributeValue), attributeName, t);
  } else if (tinfo == typeid(ONNX_NAMESPACE::GraphProto)) {
    attr.set_type(ONNX_NAMESPACE::AttributeProto::GRAPH);
    const ONNX_NAMESPACE::GraphProto &graph =
        popart::any_cast<const ONNX_NAMESPACE::GraphProto &>(attributeValue);
    auto *g = attr.mutable_g();
    *g      = graph;
  } else {
    throw error("Unsupported attribute value type {}", tinfo.name());
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const ConstVoidData &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);

  auto *t = attr.mutable_t();
  populateTensorProtoFromConstVoidData(attributeValue, attributeName, t);
}

ONNX_NAMESPACE::AttributeProto &
BuilderImpl::getNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  ONNX_NAMESPACE::AttributeProto *attr = nullptr;
  bool hasAttribute = nodeHasAttributeImpl(attr, node, attributeName);
  if (!hasAttribute) {
    throw error("Node does not have an attribute " + attributeName + ".");
  }
  return *attr;
}

int64_t
BuilderImpl::getInt64NodeAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not an integer.");
  }
  return attr.i();
}

std::vector<int64_t> BuilderImpl::getInt64VectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<int64_t> out;
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::INTS) {
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
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::FLOAT) {
    throw error("Attribute " + attributeName + " is not a float.");
  }
  return attr.f();
}

std::vector<float> BuilderImpl::getFloatVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<float> out;
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::FLOATS) {
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
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::STRING) {
    throw error("Attribute " + attributeName + " is not a string.");
  }
  return attr.s();
}

std::vector<std::string> BuilderImpl::getStringVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<std::string> out;
  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::STRINGS) {
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

  ONNX_NAMESPACE::AttributeProto &attr =
      getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != ONNX_NAMESPACE::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not a int.");
  }

  return static_cast<bool>(attr.i());
}

void BuilderImpl::removeNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  ONNX_NAMESPACE::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
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
  ONNX_NAMESPACE::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
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
  ONNX_NAMESPACE::checker::check_model(model_);

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
      throw error("Encountered ONNX opset version {}, Maximimum supported "
                  "opset is {}, minimum {} and default {}.",
                  opset.version(),
                  maxOnnxOperatorSetVersion,
                  minOnnxOperatorSetVersion,
                  onnxOperatorSetVersion);
    }
    if (opset.domain() == Domain::ai_graphcore &&
        (opset.version() < minGraphcoreOperatorSetVersion ||
         opset.version() < maxGraphcoreOperatorSetVersion)) {
      throw error("Encountered GraphCore opset version {}, Maximimum supported "
                  "opset is {}, minimum {} and default {}.",
                  opset.version(),
                  minGraphcoreOperatorSetVersion,
                  maxGraphcoreOperatorSetVersion,
                  graphcoreOperatorSetVersion);
    }

    // TODO : Need to check if we have already set the domain opsetversion
    // Record which opset is defined in the model so that we can verify that we
    // add op's from the same opset
    opsetVersions[opset.domain() == "" ? Domain::ai_onnx : opset.domain()] =
        opset.version();
  }
}

void BuilderImpl::saveModelProto(const std::string &fn) {
  // Check the model is valid.
  ONNX_NAMESPACE::checker::check_model(model_);

  io::writeModel(model_, fn);
}

void BuilderImpl::saveInitializersExternally(const std::vector<TensorId> &ids,
                                             const std::string &fn) {
  onnxutil::saveInitializersExternally(model_, ids, fn);
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

std::set<TensorId> BuilderImpl::getValidInputTensorIds() const {
  std::set<TensorId> ids;

  const ONNX_NAMESPACE::GraphProto *graph = &model_.graph();

  // Tensor IDs of parent graphs can be used in the child graph.
  if (nullptr != parent) {
    auto parentIds = parent->getValidInputTensorIds();
    ids.insert(parentIds.begin(), parentIds.end());
  }

  // Add input tensors within this graph.
  for (const auto &input : graph->input()) {
    ids.insert(input.name());
  }
  // Add output tensors within this graph.
  for (const auto &output : graph->output()) {
    ids.insert(output.name());
  }
  // Add value tensors within this graph.
  for (const auto &value_info : graph->value_info()) {
    ids.insert(value_info.name());
  }
  // Add node outputs.
  for (const ONNX_NAMESPACE::NodeProto &node : graph->node()) {
    for (const std::string &output : node.output()) {
      ids.insert(output);
    }
  }
  return ids;
}

bool BuilderImpl::isInputTensor(const TensorId &id) const {
  std::vector<TensorId> inIds = getInputTensorIds();
  return std::find(inIds.begin(), inIds.end(), id) != inIds.end();
}

bool BuilderImpl::isOutputTensor(const TensorId &id) const {
  std::vector<TensorId> outIds = getOutputTensorIds();
  return std::find(outIds.begin(), outIds.end(), id) != outIds.end();
}

bool BuilderImpl::isValueTensor(const TensorId &id) const {
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

const ONNX_NAMESPACE::ValueInfoProto &
BuilderImpl::getValueInfoProto(TensorId id) const {
  if (isInputTensor(id)) {
    const ONNX_NAMESPACE::ValueInfoProto &t =
        model_.graph().input(getInputTensorIndex(id));
    return t;
  } else if (isOutputTensor(id)) {
    const ONNX_NAMESPACE::ValueInfoProto &t =
        model_.graph().output(getOutputTensorIndex(id));
    return t;
  } else if (isValueTensor(id)) {
    const ONNX_NAMESPACE::ValueInfoProto &t =
        model_.graph().value_info(getValueTensorIndex(id));
    return t;
  } else {
    throw error("{} is not an known tensor. Must be one of (inputs:{}) "
                "(outputs:{}) (values:{})",
                id,
                getStrFromTensorIdVec(getInputTensorIds()),
                getStrFromTensorIdVec(getOutputTensorIds()),
                getStrFromTensorIdVec(getValueTensorIds()));
  }
}

bool BuilderImpl::hasTensorShape(const TensorId &id) const {
  if (isInputTensor(id) || isOutputTensor(id) || isValueTensor(id)) {
    auto &t = getValueInfoProto(id);
    if (t.type().tensor_type().shape().dim_size() > 0) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> BuilderImpl::getTensorShape(const TensorId &id) {
  std::vector<int64_t> shape;

  auto &t = getValueInfoProto(id);
  for (const auto &dim : t.type().tensor_type().shape().dim()) {
    shape.push_back(dim.dim_value());
  }
  return shape;
}

DataType BuilderImpl::getTensorDataType(const TensorId &id) {
  auto &t = getValueInfoProto(id);
  return onnxutil::getDataType(t.type().tensor_type().elem_type());
}

std::string BuilderImpl::getTensorDtypeString(const TensorId &id) {
  auto dataTypeInfo = &getDataTypeInfoMap().at(getTensorDataType(id));

  return dataTypeInfo->lcasename();
}

bool BuilderImpl::isInitializer(const TensorId &id) const {
  std::vector<std::string> initIds;
  for (const auto &initializer : model_.graph().initializer()) {
    if (initializer.name() == id) {
      return true;
    }
  }
  return false;
}

std::vector<TensorId> BuilderImpl::getTrainableTensorIds() const {
  std::vector<TensorId> result;
  for (const auto &initializer : model_.graph().initializer()) {
    result.push_back(initializer.name());
  }
  return result;
}

void BuilderImpl::setAttribute(const std::string &attribute,
                               popart::any value) {
  attributes.insert(std::make_pair(attribute, value));
}

popart::any BuilderImpl::getAttribute(const std::string &attribute) const {
  auto it = attributes.find(attribute);
  if (it != attributes.end()) {
    return it->second;
  }
  throw error("Attribute {} not found", attribute);
}

bool BuilderImpl::hasAttribute(const std::string &attribute) const {
  return attributes.find(attribute) != attributes.end();
}

void BuilderImpl::clearAttribute(const std::string &attribute) {
  attributes.erase(attribute);
}

bool BuilderImpl::hasAttribute(const std::string &attribute) {
  return attributes.find(attribute) != attributes.end();
}

popart::any BuilderImpl::getAttribute(const std::string &attribute) {
  return attributes.at(attribute);
}

void BuilderImpl::pushNameScope(const std::string &name) {
  name_scope_stack_.push_back(name);
}

void BuilderImpl::popNameScope() { name_scope_stack_.pop_back(); }

std::string BuilderImpl::getNameScope(const std::string &name) const {
  std::stringstream fullname;
  for (const auto &n : name_scope_stack_) {
    fullname << n << sNameDelimiter;
  }
  fullname << name;
  return fullname.str();
}

std::vector<TensorId>
BuilderImpl::checkpointOutput(const std::vector<TensorId> &nodeOutputNames) {
  std::vector<TensorId> results;
  for (const auto &arg : nodeOutputNames) {
    auto output = op(Onnx::Operators::Identity_1,
                     opsetVersions[Domain::ai_onnx],
                     {arg},
                     {},
                     "Checkpoint")
                      .at(0);
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(RecomputeType::Checkpoint),
                     {output});
    addNodeAttribute(sExcludePatternsAttribute,
                     std::vector<std::string>{"PreUniRepl", "PostNRepl"},
                     {output});
    results.push_back(output);
  }
  return results;
}

} // namespace popart
