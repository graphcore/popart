// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opsets.hpp>

namespace popart {

namespace {

OpDomain sanitizeDomain(const OpDomain &domain) {
  if (domain == "") {
    return Domain::ai_onnx;
  } else {
    return domain;
  }
}

std::vector<TensorId> getInputIds(const Node &node) {
  std::vector<TensorId> inputIds;
  for (int i = 0; i < node.input_size(); i++) {
    inputIds.push_back(node.input(i));
  }
  return inputIds;
}

std::vector<TensorId> getOutputIds(const Node &node) {
  std::vector<TensorId> outputIds;
  for (int i = 0; i < node.output_size(); i++) {
    outputIds.push_back(node.output(i));
  }
  return outputIds;
}

} // namespace

std::ostream &operator<<(std::ostream &os, const OperatorIdentifier &opid) {
  os << opid.domain << "." << opid.type << ":" << opid.version;
  return os;
}

const std::vector<TensorId> &OpCreatorInfo::getInputIds() const {
  if (inputIds.empty()) {
    throw internal_error(
        "No inputs ids were passed to the call to OpManager::createOp, but the "
        "op factory function for op {} is attempting to get the input ids.\n"
        "Consider checking the call to createOp and adding the input ids.",
        opid);
  } else {
    return inputIds;
  }
}

const std::vector<TensorId> &OpCreatorInfo::getOutputIds() const {
  if (outputIds.empty()) {
    throw internal_error(
        "No outputs ids were passed to the call to OpManager::createOp, but "
        "the "
        "op factory function for op {} is attempting to get the output ids.\n"
        "Consider checking the call to createOp and adding the output ids.",
        opid);
  } else {
    return outputIds;
  }
}

Tensor *OpCreatorInfo::getInputTensor(int index) const {
  auto id = inputIds.at(index);
  return settings.graph.get().getTensors().get(id);
}

TensorData *OpCreatorInfo::getInputTensorData(int index) const {
  auto id     = inputIds.at(index);
  auto &graph = settings.graph.get();

  if (!graph.getTensors().contains(id)) {
    throw error("The tensor `{}` is not defined", id);
  }

  auto *tensor = graph.getTensors().get(id);

  if (!tensor->hasTensorData()) {
    throw error("The tensor `{}` does not have data", id);
  }

  return tensor->tensorData();
}

TensorInfo &OpCreatorInfo::getInputTensorInfo(int index) const {
  return getInputTensor(index)->info;
}

OpManager &OpManager::getInstance() {
  static OpManager instance;
  return instance;
}

OpManager::OpFactoryFunc &OpManager::OpInfo::getSimpleFactory() {
  if (!simpleFactory) {
    throw internal_error("Simple factory function for op was not set.");
  }
  return *simpleFactory;
}

bool OpManager::OpInfo::hasComplexFactory() {
  return static_cast<bool>(complexFactory);
}

OpManager::ComplexOpFactoryFunc &OpManager::OpInfo::getComplexFactory() {
  if (!complexFactory) {
    throw internal_error("Complex factory function for op was not set.");
  }
  return *complexFactory;
}

void OpManager::registerOp(const OpInfo &opInfo) {
  auto opid = opInfo.id;
  auto it   = getInstance().opMap.find(std::make_pair(opid.domain, opid.type));
  if (it != getInstance().opMap.end()) {
    // Add to list
    it->second.insert(std::make_pair(opid.version, opInfo));
  } else {
    // Add new entry for domain/type
    std::map<int, OpInfo> map;
    map.insert(std::make_pair(opid.version, opInfo));
    getInstance().opMap.insert(
        std::make_pair(std::make_pair(opid.domain, opid.type), map));
  }
}

const std::vector<OperatorIdentifier>
OpManager::getSupportedOperations(bool includePrivate) {
  std::vector<OperatorIdentifier> list;

  for (auto &op : OpManager::getInstance().opMap) {
    for (auto &opVersion : op.second) {
      if (opVersion.second.isPublic || includePrivate) {
        list.push_back(opVersion.second.id);
      }
    }
  }

  return list;
}

const OpDefinitions
OpManager::getSupportedOperationsDefinition(bool includePrivate) {
  OpDefinitions list;

  for (auto &op : OpManager::getInstance().opMap) {
    for (auto &opVersion : op.second) {
      if (opVersion.second.isPublic || includePrivate) {
        list.insert({opVersion.second.id, opVersion.second.details});
      }
    }
  }

  return list;
}

Attributes OpManager::getAttributesFromAnyMap(
    std::map<std::string, popart::any> attributes) {
  Attributes attr;
  for (auto attribute : attributes) {
    const std::type_info &tinfo = attribute.second.type();
    if (tinfo == typeid(Attributes::Int)) {
      auto value = popart::any_cast<Attributes::Int>(attribute.second);
      attr.setAttribute(attribute.first, value);
    } else if (tinfo == typeid(Attributes::Ints)) {
      auto value = popart::any_cast<Attributes::Ints>(attribute.second);
      attr.setAttribute(attribute.first, value);
    } else if (tinfo == typeid(std::string)) {
      auto value = popart::any_cast<std::string>(attribute.second);
      attr.setAttribute(attribute.first, value);
    } else {
      throw error("Unsupported attribute value type {}", tinfo.name());
    }
  }
  return attr;
}

OpManager::OpInfo *OpManager::findOpInfo(const OpDomain &opDomain,
                                         const OpType &type,
                                         int opsetVersion) {
  OpDomain domain = sanitizeDomain(opDomain);
  int version     = 0;
  OpInfo *opInfo  = nullptr;

  // First find the domain/type
  auto it2 = opMap.find(std::make_pair(domain, type));
  if (it2 != opMap.end()) {
    for (auto &it3 : it2->second) {

      // Then find the op with the largest version that is less than the opset
      // version
      if (it3.first >= version && it3.first <= opsetVersion) {
        version = it3.first;
        opInfo  = &it3.second;
      }
    }
  }

  return opInfo;
}

void OpManager::checkOpVersionAgainstOpset(const OpInfo *opInfo,
                                           int opsetVersion,
                                           Graph &graph) {
  if (!opInfo) {
    return;
  }

  auto &ir = graph.getIr();

  auto domain    = opInfo->id.domain;
  auto opType    = opInfo->id.type;
  auto opVersion = opInfo->id.version;

  // Check the version we have for the op matches the version in the opset.
  if (ir.getSessionOptions().strictOpVersions && domain == Domain::ai_onnx) {
    auto opid = getOpid(domain, opsetVersion, opType);
    if (opid.version != opVersion) {
      throw error("For an opset {} Model, the ONNX spec stipulates that a {} "
                  "op must be version {}. The highest version we have "
                  "implemented less than or equal to {} is {}, so bailing.",
                  opsetVersion,
                  opType,
                  opid.version,
                  opid.version,
                  opVersion);
    }
  }
}

std::unique_ptr<Op>
OpManager::createOp(const OpDomain &opDomain,
                    const OpType &type,
                    const int opsetVersion,
                    Graph &graph,
                    const std::string &name,
                    const Scope &scope,
                    const Attributes &attr,
                    const std::vector<TensorId> &inputIds,
                    const std::vector<TensorId> &outputIds) {

  OpManager &self = getInstance();

  OpInfo *opInfo = self.findOpInfo(opDomain, type, opsetVersion);
  self.checkOpVersionAgainstOpset(opInfo, opsetVersion, graph);

  if (opInfo != nullptr) {
    return self.create(opInfo->id,
                       graph,
                       name,
                       scope,
                       attr,
                       inputIds,
                       outputIds,
                       opInfo->getSimpleFactory());
  }
  return nullptr;
}

std::unique_ptr<Op> OpManager::createOp(const OperatorIdentifier &opid,
                                        Graph &graph,
                                        const std::string &name,
                                        const Attributes &attr) {

  OpManager &self = getInstance();

  // First find the domain/type
  const auto &it2 = self.opMap.find(std::make_pair(opid.domain, opid.type));

  if (it2 != self.opMap.end()) {
    // Then find the version
    const auto &it3 = it2->second.find(opid.version);

    if (it3 != it2->second.end()) {
      return self.create(
          opid, graph, name, {}, attr, {}, {}, it3->second.getSimpleFactory());
    }
  }
  return nullptr;
}

Op *OpManager::createOpInGraph(const Node &node, Graph &graph) {
  int opsetVersion = graph.getIr().getOpSetVersionFromModel(node.domain());

  OpManager &self = getInstance();

  OpInfo *opInfo = self.findOpInfo(node.domain(), node.op_type(), opsetVersion);
  self.checkOpVersionAgainstOpset(opInfo, opsetVersion, graph);

  std::vector<TensorId> inputIds  = getInputIds(node);
  std::vector<TensorId> outputIds = getOutputIds(node);
  Op *op                          = nullptr;
  if (opInfo != nullptr) {
    if (opInfo->hasComplexFactory()) {
      op = self.create(opInfo->id,
                       graph,
                       node.name(),
                       graph.getScope(),
                       node.attribute(),
                       inputIds,
                       outputIds,
                       opInfo->getComplexFactory());
    } else {
      std::unique_ptr<Op> p = self.create(opInfo->id,
                                          graph,
                                          node.name(),
                                          graph.getScope(),
                                          node.attribute(),
                                          inputIds,
                                          outputIds,
                                          opInfo->getSimpleFactory());
      if (p) {
        auto opId = graph.moveIntoGraph(std::move(p));
        op        = graph.getOp(opId);

        graph.connectInputs(node, op->id);
        graph.connectOutputs(node, op->id);
      }
    }
  }

  if (!op) {
    if (node.op_type() == Onnx::AiOnnx::OpSet9::Constant.type) {
      throw internal_error("Constant Ops are not to be added");
    } else {
      throw error("No class for {}.{}:{}",
                  (node.domain() == "" ? Domain::ai_onnx : node.domain()),
                  node.op_type(),
                  opInfo ? opInfo->id.version : 0);
    }
  }

  return op;
}

std::unique_ptr<Op> OpManager::create(const OperatorIdentifier &opid,
                                      Graph &graph,
                                      const std::string &name,
                                      const Scope &scope,
                                      const Attributes &attr,
                                      const std::vector<TensorId> &inputIds,
                                      const std::vector<TensorId> &outputIds,
                                      OpFactoryFunc func) {

  Op::Settings settings(graph, name, scope);
  settings.setFromAttributes(attr);

  OpCreatorInfo info(opid, settings, attr, inputIds, outputIds);
  return func(info);
}

Op *OpManager::create(const OperatorIdentifier &opid,
                      Graph &graph,
                      const std::string &name,
                      const Scope &scope,
                      const Attributes &attr,
                      const std::vector<TensorId> &inputIds,
                      const std::vector<TensorId> &outputIds,
                      ComplexOpFactoryFunc func) {
  Op::Settings settings(graph, name, scope);
  settings.setFromAttributes(attr);

  OpCreatorInfo info(opid, settings, attr, inputIds, outputIds);
  return func(info, graph);
}

OpVersion OpManager::getOpVersionFromOpSet(const OpDomain &opDomain,
                                           const OpType &type,
                                           const int opsetVersion) {
  OpManager &self = getInstance();

  OpDomain domain = opDomain;
  if (domain == "")
    domain = Domain::ai_onnx;

  int version = 0;

  // First find the domain/type
  auto it2 = self.opMap.find(std::make_pair(domain, type));
  if (it2 != self.opMap.end()) {
    for (auto &it3 : it2->second) {

      // Then find the op with the largest version that is less than the opset
      // version
      if (it3.first > version && it3.first <= opsetVersion) {
        version = it3.first;
      }
    }
  }

  return version;
}

std::ostream &operator<<(std::ostream &os,
                         const std::vector<DataType> &dataTypes) {

  for (auto &dt : dataTypes) {

    if (dt != dataTypes[0]) {
      os << ", ";
    }

    os << "tensor(" << dt << ")";
  }

  return os;
}

} // namespace popart
