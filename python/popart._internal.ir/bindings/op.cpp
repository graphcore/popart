// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op.hpp"
#include "bindings/ir.hpp"

#include <iostream>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <popart/basicoptionals.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opserialiser.hpp>
#include <popart/shardingplan.hpp>
#include <popart/vertex.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindOp(py::module &m) {

  py::class_<OpDebugInfo>(m, "OpDebugInfo", py::module_local())
      .def(py::init<const DebugContext &, const Op &>())
      .def("finalize", &OpDebugInfo::finalize);
  py::class_<TensorLocation>(m, "TensorLocation", py::module_local())
      .def(py::init<>())
      .def("operator=", &TensorLocation::operator=)
      .def("operator==", &TensorLocation::operator==)
      .def("operator!=", &TensorLocation::operator!=)
      .def("serialize", &TensorLocation::serialize)
      .def("isRemote", &TensorLocation::isRemote);
  py::class_<OptionalVGraphId>(m, "OptionalVGraphId")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalVGraphId::operator bool)
      .def("reset", &OptionalVGraphId::reset);
  py::class_<OptionalPipelineStage>(m, "OptionalPipelineStage")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalPipelineStage::operator bool)
      .def("reset", &OptionalPipelineStage::reset);
  py::class_<OptionalExecutionPhase>(m, "OptionalExecutionPhase")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalExecutionPhase::operator bool)
      .def("reset", &OptionalExecutionPhase::reset);
  py::class_<OptionalBatchSerializedPhase>(m, "OptionalBatchSerializedPhase")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalBatchSerializedPhase::operator bool)
      .def("reset", &OptionalBatchSerializedPhase::reset);
  py::class_<OptionalTensorLocation>(m, "OptionalTensorLocation")
      .def(py::init<>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalTensorLocation::operator bool)
      .def("reset", &OptionalTensorLocation::reset);

  py::class_<NumInputs>(m, "NumInputs")
      .def(py::init<>())
      .def(py::init<int>())
      .def(py::init<int, int>())
      .def_readwrite("min", &NumInputs::min)
      .def_readwrite("max", &NumInputs::min);
  py::class_<OperatorIdentifier>(m, "OperatorIdentifier", py::module_local())
      .def(py::init<const OpDomain, const OpType, OpVersion, NumInputs, int>(),
           py::arg("domain"),
           py::arg("type"),
           py::arg("version"),
           py::arg("inputs")  = NumInputs(),
           py::arg("outputs") = 0)
      .def_readonly("domain", &OperatorIdentifier::domain)
      .def_readonly("type", &OperatorIdentifier::type)
      .def_readonly("version", &OperatorIdentifier::version)
      .def_readonly("numInputs", &OperatorIdentifier::numInputs)
      .def_readonly("numOutputs", &OperatorIdentifier::numOutputs)
      .def(py::self < py::self)
      .def(py::self == py::self)
      .def(py::self != py::self);
  py::class_<Op::Settings>(m, "Settings", py::module_local())
      .def(py::init<Graph &, const std::string &>())
      .def(py::init<Graph &, const std::string &, const Scope &>())
      .def("copy", &Op::Settings::copy)
      // Not binding setFromAttributes as it is ONNX based.
      .def_readwrite("name", &Op::Settings::name)
      .def("getIr", &Op::Settings::getIr);
  py::class_<Op, PyOp<>, std::shared_ptr<Op>>(m, "Op")
      .def(py::init<const OperatorIdentifier &, const Op::Settings &>())
      .def_readwrite("id", &Op::id)
      .def_readwrite("opid", &Op::opid)
      .def("getSettings", py::overload_cast<>(&Op::getSettings))
      .def("getInSettings", &Op::getInSettings)
      .def("getOutSettings", &Op::getOutSettings)
      .def("adjustInSettings", &Op::adjustInSettings)
      .def("adjustOutSettings", &Op::adjustOutSettings)
      .def("getOptionalVGraphId", &Op::getOptionalVGraphId)
      .def("getVirtualGraphId", &Op::getVirtualGraphId)
      .def("getIntrospectionInVirtualGraphId",
           py::overload_cast<InIndex>(&Op::getIntrospectionInVirtualGraphId,
                                      py::const_))
      .def("getIntrospectionInVirtualGraphId",
           py::overload_cast<InIndex, std::set<OpId> &>(
               &Op::getIntrospectionInVirtualGraphId, py::const_))
      .def("setVirtualGraphId", &Op::setVirtualGraphId)
      .def("hasVirtualGraphId", &Op::hasVirtualGraphId)
      .def("getOptionalExecutionPhase", &Op::getOptionalExecutionPhase)
      .def("getExecutionPhase", &Op::getExecutionPhase)
      .def("setExecutionPhase", &Op::setExecutionPhase)
      .def("hasExecutionPhase", &Op::hasExecutionPhase)
      .def("getOptionalBatchSerializedPhase",
           &Op::getOptionalBatchSerializedPhase)
      .def("getBatchSerializedPhase", &Op::getBatchSerializedPhase)
      .def("setBatchSerializedPhase", &Op::setBatchSerializedPhase)
      .def("hasBatchSerializedPhase", &Op::hasBatchSerializedPhase)
      .def("isExcludedFromPattern", &Op::isExcludedFromPattern)
      .def("setPipelineStage", &Op::setPipelineStage)
      .def("hasPipelineStage", &Op::hasPipelineStage)
      .def("getPipelineStage", &Op::getPipelineStage)
      .def("getOptionalPipelineStage", &Op::getOptionalPipelineStage)
      .def("getInBatchAxis", &Op::getInBatchAxis)
      .def("getOutBatchAxis", &Op::getOutBatchAxis)
      // TODO: T42791 alias related methods
      // .def("inheritPlacementAttributes", &Op::inheritPlacementAttributes)
      .def("getIr", py::overload_cast<>(&Op::getIr))
      .def("getGraph",
           py::overload_cast<>(&Op::getGraph),
           py::return_value_policy::reference)
      .def("getScope", &Op::getScope)
      .def("setScope", &Op::setScope)
      .def("getName", &Op::getName)
      .def("setName", &Op::setName)
      .def(
          "getDebugInfo", &Op::getDebugInfo, py::return_value_policy::reference)
      .def("isNorm", &Op::isNorm)
      .def("isElementWiseUnary", &Op::isElementWiseUnary)
      .def("canBeReplacedByIdentity", &Op::canBeReplacedByIdentity)
      .def("str", &Op::str)
      .def("debugName", &Op::debugName)
      .def("createAndConnectOutTensor", &Op::createAndConnectOutTensor)
      // Stringstream stuff not required
      // .def("append", &Op::append)
      // .def("toJSON", &Op::toJSON)
      .def("memOfOutputs", &Op::memOfOutputs)
      // TODO: T41718 op specific tests
      .def("optionalInputs", &Op::optionalInputs)
      .def("defaultConnectInTensor", &Op::defaultConnectInTensor)
      .def("connectInTensor", &Op::connectInTensor)   // virtual
      .def("connectOutTensor", &Op::connectOutTensor) // virtual
      .def("disconnectInTensor",
           py::overload_cast<Tensor *>(&Op::disconnectInTensor))
      .def("disconnectInTensor",
           py::overload_cast<InIndex>(&Op::disconnectInTensor))
      .def("disconnectOutTensor", &Op::disconnectOutTensor)
      .def("disconnectAllInputs", &Op::disconnectAllInputs)
      .def("disconnectAllOutputs", &Op::disconnectAllOutputs)
      .def("name", &Op::name)
      // TODO: T41718 op specific tests
      .def("setup", &Op::setup)
      // See above for why it is PyOp<>::clone_wrapper
      .def("clone", [](PyOp<> &self) { return self.clone_wrapper(); })
      .def("getSubgraphValue", &Op::getSubgraphValue)
      .def("finalizeDebugInfo", &Op::finalizeDebugInfo)
      .def("setCalledSubgraphGradInfo", &Op::setCalledSubgraphGradInfo)
      .def("getGradOps",
           [](Op &self) {
             auto uniqueVec = self.getGradOps();
             std::vector<std::shared_ptr<Op>> sharedVec;
             std::move(uniqueVec.begin(),
                       uniqueVec.end(),
                       std::back_inserter(sharedVec));
             return sharedVec;
           })
      .def("inplacePriorityDefault", &Op::inplacePriorityDefault)
      // TODO: T41718 op specific tests
      .def("getInplaceVariant",
           [](PyOp<> &self, const OperatorIdentifier &opid) {
             return self.getInplaceVariant_wrapper(opid);
           })
      // TODO: T42791 alias related methods
      //  .def("modifies", &Op::modifies)
      //  .def("uses", &Op::uses)
      //  .def("aliases", &Op::aliases)
      //  .def("fwdRegMap", &Op::fwdRegMap)
      //  .def("bwdRegMap", &Op::bwdRegMap)
      //  .def("doesAlias", &Op::doesAlias)
      //  .def("isOutplace", &Op::isOutplace)
      //  .def("doesAlias", &Op::doesAlias)
      //  .def("modifies", &Op::modifies)
      .def("modifiesIndex", &Op::modifiesIndex)
      .def("overwritesTensor", &Op::overwritesTensor)
      .def("isInplaceViewChange", &Op::isInplaceViewChange)
      .def("isOutplaceViewChange", &Op::isOutplaceViewChange)
      .def("getNonGradInIndex", &Op::getNonGradInIndex)
      .def("gradInputInfo", &Op::gradInputInfo)
      .def("gradOutToNonGradIn", &Op::gradOutToNonGradIn)
      .def("isLossOp", &Op::isLossOp)
      .def("isIpuCopyOp", &Op::isIpuCopyOp)
      .def("copiesOptimizerTensors", &Op::copiesOptimizerTensors)
      .def("isOptimizerOp", &Op::isOptimizerOp)
      .def("requiresRandomSeed", &Op::requiresRandomSeed)
      .def("getSeedInIndex", &Op::getSeedInIndex)
      .def("hasInput", &Op::hasInput)
      .def("hasOutput", &Op::hasOutput)
      .def("inTensor",
           py::overload_cast<InIndex>(&Op::inTensor),
           py::return_value_policy::reference)
      .def("outTensor",
           py::overload_cast<OutIndex>(&Op::outTensor),
           py::return_value_policy::reference)
      .def("inId", py::overload_cast<InIndex>(&Op::inId))
      .def("outId", py::overload_cast<OutIndex>(&Op::outId))
      .def("inInfo",
           py::overload_cast<InIndex>(&Op::inInfo),
           py::return_value_policy::reference)
      .def("outInfo",
           py::overload_cast<OutIndex>(&Op::outInfo),
           py::return_value_policy::reference)
      .def("inShape", &Op::inShape)
      .def("outShape", &Op::outShape)
      .def("inTensorCount", &Op::inTensorCount)
      .def("outTensorCount", &Op::outTensorCount)
      .def("inRank", &Op::inRank)
      .def("outRank", &Op::outRank)
      .def("outIndex", &Op::outIndex)
      // TODO: T42819 add bindings for (outline) attributes
      //   .def("appendAttributes", &Op::appendAttributes)
      //   .def("appendOutlineAttributes", &Op::appendOutlineAttributes)
      .def("prettyNpOut",
           py::overload_cast<const Shape &, const Shape &>(&Op::prettyNpOut,
                                                           py::const_))
      .def("prettyNpOut",
           py::overload_cast<const TensorInfo &, const TensorInfo &>(
               &Op::prettyNpOut, py::const_))
      // TODO: T41718 test call ops
      .def("getCalledGraphs", &Op::getCalledGraphs)
      .def("getCalledGraphIds", &Op::getCalledGraphIds)
      .def("getCalledGraphIndex", &Op::getCalledGraphIndex)
      .def("opInToSubgraphInIndex", &Op::opInToSubgraphInIndex)
      .def("subgraphInToOpInIndex", &Op::subgraphInToOpInIndex)
      .def("opOutToSubgraphOutIndex", &Op::opOutToSubgraphOutIndex)
      .def("subgraphOutToOpOutIndex", &Op::subgraphOutToOpOutIndex)
      .def("getSubgraphEquivId", &Op::getSubgraphEquivId)
      .def("getSubgraphInputs", &Op::getSubgraphInputs)
      .def("getSubgraphOutputs", &Op::getSubgraphOutputs)
      .def("getHighSubgraphValue", &Op::getHighSubgraphValue)
      .def("getLowSubgraphValue", &Op::getLowSubgraphValue)
      .def("calcAutoVirtualGraphCost", &Op::calcAutoVirtualGraphCost)
      .def("isOutlineable", &Op::isOutlineable)
      .def("hasSideEffect", &Op::hasSideEffect)
      .def("inputsUnmodifiable", &Op::inputsUnmodifiable)
      .def("consumesGraphOutput", &Op::consumesGraphOutput)
      .def("producesGraphOutput", &Op::producesGraphOutput)
      .def("inputUnmodifiable", &Op::inputUnmodifiable)
      .def("hasAliasedModifiers", &Op::hasAliasedModifiers)
      .def("isParentOf", &Op::isParentOf)
      .def("isChildOf", &Op::isChildOf)
      .def("canShard", &Op::canShard)
      // TODO: T42793 sharding methods
      //   .def("getShardReductionType", &Op::getShardReductionType)
      //   .def("getShardRescaleFactor", &Op::getShardRescaleFactor)
      // .def("shard", py::overload_cast<const ShardingPlan>(&Op::shard))
      // .def("shard",
      //      py::overload_cast<const std::map<TensorId, std::vector<TensorId>>
      //      &>(
      //          &Op::shard))
      // .def("configureShardedOp", &Op::configureShardedOp)
      // .def("getReplicatedTensorShardingIndices",
      //      &Op::getReplicatedTensorShardingIndices)
      // .def("configureForReplicatedTensorSharding",
      //      &Op::configureForReplicatedTensorSharding)
      .def("transferBaseProperties", &Op::transferBaseProperties);
} // namespace ir

} // namespace ir
} // namespace _internal
} // namespace popart
