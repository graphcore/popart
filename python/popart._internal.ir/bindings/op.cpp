// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op.hpp"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <pybind11/attr.h>
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <set>
#include <string>
#include <utility>
#include <popart/basicoptionals.hpp>
#include <popart/graph.hpp>
#include <popart/op.hpp>

#include "popart/error.hpp"
#include "popart/ir.hpp" // IWYU pragma: keep
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/scope.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/vendored/any.hpp" // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindOp(py::module &m) {
  py::class_<Op::Settings>(m, "Settings", py::module_local())
      .def(py::init<Graph &, const std::string &>())
      .def(py::init<Graph &, const std::string &, const Scope &>())
      .def("copy", &Op::Settings::copy)
      // Not binding setFromAttributes as it is ONNX based.
      .def_readwrite("name", &Op::Settings::name)
      .def("getIr", &Op::Settings::getIr)
      .def_readwrite("vgraphId", &Op::Settings::vgraphId)
      .def_readwrite("pipelineStage", &Op::Settings::pipelineStage)
      .def_readwrite("inferTensorMappingToFrom",
                     &Op::Settings::inferTensorMappingToFrom)
      .def_readwrite("debugInfoId", &Op::Settings::debugInfoId)
      .def_readwrite("executionContext", &Op::Settings::executionContext)
      .def_readwrite("tileSet", &Op::Settings::tileSet);

  py::enum_<ExecutionContext>(m, "ExecutionContext", py::module_local())
      .value("Normal", ExecutionContext::Normal)
      .value("AccumulateOuterFragment",
             ExecutionContext::AccumulateOuterFragment)
      .value("WeightsFromHostFragment",
             ExecutionContext::WeightsFromHostFragment)
      .value("WeightsToHostFragment", ExecutionContext::WeightsToHostFragment)
      .value("OptimizerFromHostFragment",
             ExecutionContext::OptimizerFromHostFragment)
      .value("Subgraph", ExecutionContext::Subgraph);

  py::class_<Op, PyOp<>, std::shared_ptr<Op>>(m, "Op")
      .def(py::init<const OperatorIdentifier &, const Op::Settings &>())
      .def_readwrite("id", &Op::id)
      .def_readwrite("opid", &Op::opid)
      // Bind a method that returns the OpType only, as we do not want to deal
      // with full ONNX OpIdentifiers in the high-level Python API.
      .def("opType",
           [](const Op &self) -> std::string { return self.opid.type; })
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
      .def("optionalInputs", &Op::optionalInputs)
      .def("defaultConnectInTensor", &Op::defaultConnectInTensor)
      .def(
          "connectInTensor",
          py::overload_cast<InIndex, TensorId>(&Op::connectInTensor)) // virtual
      .def("connectOutTensor", &Op::connectOutTensor)                 // virtual
      .def("disconnectInTensor",
           py::overload_cast<Tensor *>(&Op::disconnectInTensor))
      .def("disconnectInTensor",
           py::overload_cast<InIndex>(&Op::disconnectInTensor))
      .def("disconnectOutTensor", &Op::disconnectOutTensor)
      .def("disconnectAllInputs", &Op::disconnectAllInputs)
      .def("disconnectAllOutputs", &Op::disconnectAllOutputs)
      .def("name", &Op::name)
      .def("setup", &Op::setup)
      // See above for why it is PyOp<>::clone_wrapper
      .def("clone", [](PyOp<> &self) { return self.clone_wrapper(); })
      .def(
          "cloneIntoGraph",
          [](Op &self, Graph &graph) {
            // Clone the operator
            auto clonedOpUp = self.clone();
            // Change ownership of the cloned operator after obtaining the raw
            // pointer
            auto clonedOp = clonedOpUp.get();
            graph.moveIntoGraph(std::move(clonedOpUp));
            // Change scope of the clonedOp so that it is no longer a part of
            // the old graph
            clonedOp->settings.scope = graph.getScope();
            return clonedOp;
          },
          py::return_value_policy::reference)
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
      .def("getInplaceVariant",
           [](PyOp<> &self, const OperatorIdentifier &opid) {
             return self.getInplaceVariant_wrapper(opid);
           })
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
      .def(
          "hasInputTensor",
          [](Op &self, Tensor *tensor) { return self.input->contains(tensor); })
      .def("inTensor",
           py::overload_cast<InIndex>(&Op::inTensor),
           py::return_value_policy::reference)
      .def("outTensor",
           py::overload_cast<OutIndex>(&Op::outTensor),
           py::return_value_policy::reference)
      .def(
          "getInputTensors",
          [](Op &self) {
            // Return elements in index order
            std::vector<Tensor *> inputs;
            for (auto &idx_tensor : self.input->tensorMap()) {
              inputs.push_back(idx_tensor.second);
            }
            return inputs;
          },
          py::return_value_policy::reference)
      .def(
          "getOutputTensors",
          [](Op &self) {
            // Return elements in index order
            std::vector<Tensor *> outputs;
            for (auto &idx_tensor : self.output->tensorMap()) {
              outputs.push_back(idx_tensor.second);
            }
            return outputs;
          },
          py::return_value_policy::reference)
      .def(
          "getInputIndexMap",
          [](Op &self) { return self.input->tensorMap(); },
          py::return_value_policy::reference)
      .def(
          "getOutputIndexMap",
          [](Op &self) { return self.output->tensorMap(); },
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
      .def("inIndex", &Op::inIndex)
      .def("outIndex", &Op::outIndex)
      .def("firstInIndex",
           [](const Op *self, Tensor *t) -> InIndex {
             const auto indices = self->input->indices(t);
             if (indices.size() < 1) {
               throw popart::error(
                   "Op `{}` does not have Tensor `{}` as an input",
                   self->debugName(),
                   t->id);
             }
             return indices[0];
           })
      .def("prettyNpOut",
           py::overload_cast<const Shape &, const Shape &>(&Op::prettyNpOut,
                                                           py::const_))
      .def("prettyNpOut",
           py::overload_cast<const TensorInfo &, const TensorInfo &, bool>(
               &Op::prettyNpOut, py::const_),
           py::arg("s0"),
           py::arg("s1"),
           py::arg("checkDataType") = true)
      .def("getCalledGraphs",
           &Op::getCalledGraphs,
           py::return_value_policy::reference)
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
      .def("transferBaseProperties", &Op::transferBaseProperties);
} // namespace ir

} // namespace ir
} // namespace _internal
} // namespace popart
