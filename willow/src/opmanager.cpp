// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <ostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>
#include <popart/debugcontext.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxdebuginfo.hpp>
#include <popart/opmanager.hpp>
#include <popart/opsets.hpp>
#include <popart/util.hpp>

#include "popart/attributes.hpp"
#include "popart/basicoptionals.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/any.hpp"

namespace popart {
class Scope;
class TensorData;

namespace {

OpDomain sanitizeDomain(const OpDomain &domain) {
  if (domain == "") {
    return Domain::ai_onnx;
  } else {
    return domain;
  }
}

std::vector<TensorId> getInputIds(const Node &node, const Graph &graph) {
  std::vector<TensorId> inputIds;
  for (int i = 0; i < node.input_size(); i++) {
    inputIds.push_back(addScope(graph, node.input(i)));
  }
  return inputIds;
}

std::vector<TensorId> getOutputIds(const Node &node, const Graph &graph) {
  std::vector<TensorId> outputIds;
  for (int i = 0; i < node.output_size(); i++) {
    outputIds.push_back(addScope(graph, node.output(i)));
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

bool OpCreatorInfo::hasInputTensor(int index) const {
  if (index >= inputIds.size()) {
    return false;
  } else {
    auto id = inputIds.at(index);
    return settings.graph.get().getTensors().contains(id);
  }
}

// Returns formatted information about the op.
// Similar to Op::debugName
std::string OpCreatorInfo::debugName() const {
  std::string debug_id = settings.name + "[" + opid.domain + "." + opid.type +
                         ":" + std::to_string(opid.version) + "]";

  std::vector<TensorId> in_ids  = getInputIds();
  std::vector<TensorId> out_ids = getOutputIds();

  return logging::format("Op({}, inputs=[{}], outputs=[{}])",
                         debug_id,
                         logging::join(in_ids.begin(), in_ids.end(), ", "),
                         logging::join(out_ids.begin(), out_ids.end(), ", "));
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

const std::vector<OperatorIdentifier>
OpManager::getUnsupportedOperations(int opsetVersion) {
  std::vector<OperatorIdentifier> result;

  for (auto &op : getOpset(opsetVersion)) {
    bool foundOp = [&]() {
      auto foundType = getInstance().opMap.find({Domain::ai_onnx, op.type});
      if (foundType == getInstance().opMap.end()) {
        return false;
      }

      auto &versions    = foundType->second;
      auto foundVersion = versions.find(op.version);
      return foundVersion != versions.end();
    }();

    if (!foundOp) {
      result.push_back(op);
    }
  }

  return result;
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
    OperatorIdentifier opid = [&]() {
      try {
        return getOpid(domain, opsetVersion, opType);
      } catch (internal_error &err) {
        throw error("Internal error encounterd when checking op version "
                    "against opset.\n{}This check may be disabled by setting "
                    "popart::SessionOptions::strictOpVersions to false.",
                    err.what());
      }
    }();

    if (opid.version != opVersion) {
      throw error("For an opset {} Model, the ONNX spec stipulates that a {} "
                  "op must be version {}. The highest version we have "
                  "implemented less than or equal to {} is {}, so bailing. "
                  "This check may be disabled by setting "
                  "popart::SessionOptions::strictOpVersions to false.",
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
  return createOpWithInputs(opid, graph, name, attr, {});
}

std::unique_ptr<Op>
OpManager::createOpWithInputs(const OperatorIdentifier &opid,
                              Graph &graph,
                              const std::string &name,
                              const Attributes &attr,
                              const std::vector<TensorId> &inIds) {

  OpManager &self = getInstance();

  // First find the domain/type
  const auto &it2 = self.opMap.find(std::make_pair(opid.domain, opid.type));

  if (it2 != self.opMap.end()) {
    // Then find the version
    const auto &it3 = it2->second.find(opid.version);

    if (it3 != it2->second.end()) {
      return self.create(opid,
                         graph,
                         name,
                         {},
                         attr,
                         inIds,
                         {},
                         it3->second.getSimpleFactory());
    }
  }
  return nullptr;
}

Op *OpManager::createOpInGraph(const Node &node, Graph &graph) {
  int opsetVersion = graph.getIr().getOpSetVersionFromModel(node.domain());

  OpManager &self = getInstance();

  OpInfo *opInfo = self.findOpInfo(node.domain(), node.op_type(), opsetVersion);
  self.checkOpVersionAgainstOpset(opInfo, opsetVersion, graph);

  std::vector<TensorId> inputIds  = getInputIds(node, graph);
  std::vector<TensorId> outputIds = getOutputIds(node, graph);
  Op *op                          = nullptr;

  // Find the debugInfoId attribute if is exists. Otherwise 0
  std::uint64_t debugInfoId = 0;
  for (auto &attribute : node.attribute()) {
    auto name = attribute.name();
    if (name == sDebugInfoId) {
      debugInfoId = attribute.i();
    }
  }

  // Create the OnnxOpDebugInfo
  DebugNameAndId dnai(debugInfoId, node.name());
  OnnxOpDebugInfo odi({dnai}, node);

  // Replace the parent debugInfoId with the OnnxDebugInfo id
  popart::Attributes attributes = node.attribute();
  // Warning : converting uint64_t to int64_t;
  Attributes::Int id = odi.getId();
  attributes.setAttribute(sDebugInfoId, id);

  if (opInfo != nullptr) {
    if (opInfo->hasComplexFactory()) {
      op = self.create(opInfo->id,
                       graph,
                       node.name(),
                       graph.getScope(),
                       attributes,
                       inputIds,
                       outputIds,
                       opInfo->getComplexFactory());
    } else {
      std::unique_ptr<Op> p = self.create(opInfo->id,
                                          graph,
                                          node.name(),
                                          graph.getScope(),
                                          attributes,
                                          inputIds,
                                          outputIds,
                                          opInfo->getSimpleFactory());
      if (p) {
        auto opId = graph.moveIntoGraph(std::move(p));
        op        = graph.getOp(opId);

        graph.connectInputs(node, op->id);
        graph.connectOutputs(node, op->id);

        // Remove empty constant inputs
        std::vector<int> emptyIndex;
        const auto input_count = op->inTensorCount();
        for (int i = 0, num = 0; num < input_count; i++) {
          if (op->hasInput(i)) {
            if (op->inInfo(i).nelms() == 0 &&
                op->inTensor(i)->tensorType() == TensorType::Const) {
              emptyIndex.push_back(i);
            }
            num++;
          }
        }
        if (!emptyIndex.empty()) {
          // Disconnect empty inputs and reconnect the rest
          std::vector<TensorId> allInputs;
          for (int i = 0, num = 0; num < input_count; i++) {
            if (op->hasInput(i)) {
              allInputs.push_back(op->inTensor(i)->id);
              num++;
            }
          }
          op->disconnectAllInputs();
          int new_idx = 0;
          for (int i = 0; i < input_count; i++) {
            auto iter = std::find(emptyIndex.begin(), emptyIndex.end(), i);
            if (iter == emptyIndex.end()) {
              op->connectInTensor(new_idx, allInputs[i]);
              new_idx++;
            }
          }
        }
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

  DebugInfo di({"create"}, "popartbuilder");
  Op::Settings settings(graph, name, scope, di.getId());
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
  DebugInfo di({"create"}, "popartbuilder");
  Op::Settings settings(graph, name, scope, di.getId());
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
