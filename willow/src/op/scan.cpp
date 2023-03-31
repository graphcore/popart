// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/scan.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/subgraph.hpp"
#include "popart/operators.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

ScanOp::ScanOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_,
               int numScanInputs_,
               int numImplicitInputs_,
               std::vector<int64_t> scanInputAxes_,
               std::vector<int64_t> scanInputDirections_,
               std::vector<int64_t> scanOutputAxes_,
               std::vector<int64_t> scanOutputDirections_)
    : SubgraphOp(_opid, settings_), callee(callee_),
      numImplicitInputs(numImplicitInputs_), numScanInputs(numScanInputs_),
      baseScanInputAxes(scanInputAxes_),
      baseScanInputDirections(scanInputDirections_),
      baseScanOutputAxes(scanOutputAxes_),
      baseScanOutputDirections(scanOutputDirections_) {}

std::unique_ptr<Op> ScanOp::clone() const {
  return std::make_unique<ScanOp>(*this);
}

Graph &ScanOp::getCalledGraph() const { return callee.get(); }

void ScanOp::setCalledGraph(Graph &graph) { callee = graph; }

InIndex ScanOp::subgraphInToOpInIndex(InIndex index) const { return index; }

InIndex ScanOp::opInToSubgraphInIndex(InIndex index) const { return index; }

OutIndex ScanOp::subgraphOutToOpOutIndex(OutIndex index) const { return index; }
OutIndex ScanOp::opOutToSubgraphOutIndex(OutIndex index) const { return index; }

void ScanOp::setup() {
  auto M = getNumScanInputs();
  auto N = getNumVariables();
  auto K = getNumScanOutputs();
  auto L = getNumImplicitInputs();

  logging::op::trace("[ScanOp] {} variables, {} scan inputs, {} implicit "
                     "inputs, {} scan outputs",
                     N,
                     M,
                     L,
                     K);

  // Check inputs, calculate the number of iterations
  for (int i = 0; i < M; ++i) {
    auto iterations = inInfo(N + i).shape().at(getScanInputAxis(i));
    if (getTripCountValue() != iterations) {
      throw error("[ScanOp] Number of iterations required by the inputs do not "
                  "match: {} {}",
                  getTripCountValue(),
                  iterations);
    }
  }

  // Output shapes 0, .., N-1
  for (int n = 0; n < N; ++n) {
    outInfo(n) = inInfo(n);
    logging::op::trace("[ScanOp] Output {} info {}", n, outInfo(n));
  }

  // Output shapes N, .., N+K-1
  for (int k = 0; k < K; ++k) {
    auto tensorId = getCalledGraph().getOutputId(N + k);
    auto tensor   = getCalledGraph().getTensors().get(tensorId);
    auto info     = tensor->info;
    auto shape    = info.shape();
    shape[getScanOutputAxis(k)] *= getTripCountValue();
    info.set(info.dataType(), shape, info.metaShape());
    outInfo(N + k) = info;
    logging::op::trace("[ScanOp] Output {} info {}", N + k, outInfo(N + k));
  }
}

int ScanOp::getTripCountValue() const {
  auto N = getNumVariables();
  return inInfo(N).shape().at(getScanInputAxis(0));
}

int64_t ScanOp::getScanOutputAxis(int i) const {
  int64_t v = 0;
  if (i < baseScanOutputAxes.size()) {
    v = baseScanOutputAxes[i];
  }

  auto N = getNumVariables();

  auto tensorId = getCalledGraph().getOutputId(N + i);
  auto tensor   = getCalledGraph().getTensors().get(tensorId);
  auto rank     = tensor->info.rank();
  // Adjust output axis
  return (v + rank) % rank;
}

int64_t ScanOp::getScanInputAxis(int i) const {
  if (baseScanInputAxes.size() > i) {
    auto N    = getNumVariables();
    auto rank = inInfo(N + i).rank();
    return (baseScanInputAxes.at(i) + rank) % rank;
  } else {
    return 0;
  }
}

int64_t ScanOp::getScanInputDirection(int i) const {
  if (i < baseScanInputDirections.size()) {
    return baseScanInputDirections.at(i);
  } else {
    return 0;
  }
}

int64_t ScanOp::getScanOutputDirection(int i) const {
  if (i < baseScanOutputDirections.size()) {
    return baseScanOutputDirections.at(i);
  } else {
    return 0;
  }
}

void ScanOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
  os.appendAttribute("getTripCountValue", getTripCountValue());
  os.appendAttribute("numImplicitInputs", numImplicitInputs);
  os.appendAttribute("numScanInputs", numScanInputs);
  os.appendAttribute("baseScanInputAxes", baseScanInputAxes);
  os.appendAttribute("baseScanInputDirections", baseScanInputDirections);
  os.appendAttribute("baseScanOutputAxes", baseScanOutputAxes);
  os.appendAttribute("baseScanOutputDirections", baseScanOutputDirections);
}

namespace {

static OpDefinition::DataTypes V = {DataType::BOOL,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition scanOpDef({OpDefinition::Inputs({{"inputs", V}}),
                               OpDefinition::Outputs({{"outputs", V}}),
                               OpDefinition::Attributes({
                                   {"body", {"*"}},
                                   {"num_scan_inputs", {"*"}},
                                   {"scan_input_axes", {"*"}},
                                   {"scan_input_directions", {"*"}},
                                   {"scan_output_axes", {"*"}},
                                   {"scan_output_directions", {"*"}},
                               })});

static OpCreator<ScanOp> scanOpCreator(
    OpDefinitions({{Onnx::Operators::Scan_9, scanOpDef},
                   {Onnx::Operators::Scan_11, scanOpDef}}),
    [](const OpCreatorInfo &info, Graph &graph) -> Op * {
      // Parse attributes
      auto numScanInputs =
          info.attributes.getAttribute<Attributes::Int>("num_scan_inputs");
      auto scanInputAxes =
          info.attributes.getAttribute<Attributes::Ints>("scan_input_axes", {});
      auto scanInputDirections = info.attributes.getAttribute<Attributes::Ints>(
          "scan_input_directions", {});
      auto scanOutputAxes = info.attributes.getAttribute<Attributes::Ints>(
          "scan_output_axes", {});
      auto scanOutputDirections =
          info.attributes.getAttribute<Attributes::Ints>(
              "scan_output_directions", {});
      const ONNX_NAMESPACE::GraphProto &callee =
          info.attributes.getAttribute<Attributes::Graph>("body");

      auto &parentGraph = info.settings.graph.get();
      auto &tensors     = parentGraph.getTensors();

      auto scanBodyInputs  = SubgraphOp::getBodyInputIds(callee);
      auto scanBodyOutputs = SubgraphOp::getBodyOutputIds(callee);

      std::vector<std::pair<TensorId, TensorInfo>> opInputs;

      if (info.hasInputIds()) {
        for (int i = 0; i < info.getInputIds().size(); ++i) {
          logging::op::trace(
              "[ScanOp] Op input: {} - {}", i, info.getInputIds().at(i));
          opInputs.push_back({info.getInputIds().at(i),
                              tensors.get(info.getInputIds().at(i))->info});
        }
      }

      int64_t numVariables = (opInputs.size() - numScanInputs);

      scanInputAxes.resize(numScanInputs, 0);

      std::vector<TensorId> parentScopedImplicitTensorIds;
      auto implicitTensorIds = onnxutil::getImplicitTensorIds(callee);
      for (auto implicitTensorId : implicitTensorIds) {
        auto parentScopedImplicitTensorId =
            addScope(parentGraph, implicitTensorId);
        Tensor *tensor =
            parentGraph.getTensors().get(parentScopedImplicitTensorId);
        if (std::find(parentScopedImplicitTensorIds.begin(),
                      parentScopedImplicitTensorIds.end(),
                      parentScopedImplicitTensorId) ==
            parentScopedImplicitTensorIds.end()) {
          opInputs.push_back({implicitTensorId, tensor->info});
          parentScopedImplicitTensorIds.push_back(parentScopedImplicitTensorId);
        }
      }

      logging::op::trace("[ScanOp] Callee: {}, implicit tensors: {}",
                         callee.name(),
                         implicitTensorIds);

      auto subgraphId =
          callee.name().empty()
              ? parentGraph.getIr().createUniqueSubgraphId({"scan"})
              : callee.name();
      auto &ir          = parentGraph.getIr();
      auto &calleeGraph = ir.createGraph(subgraphId);

      int64_t numImplicit = implicitTensorIds.size();

      logging::op::trace("[ScanOp] Adding {} variables, {} scan inputs, {} "
                         "implicit inputs ({} total)",
                         numVariables,
                         numScanInputs,
                         numImplicit,
                         opInputs.size());
      for (int64_t i = 0; i < opInputs.size(); ++i) {
        auto &kv = opInputs.at(i);
        TensorId scopedTensorId;
        if (i < scanBodyInputs.size()) {
          // N state variables, M scan inputs
          scopedTensorId = addScope(calleeGraph, scanBodyInputs.at(i));
        } else {
          // L implicit inputs
          scopedTensorId = addScope(calleeGraph, kv.first);
        }
        logging::op::trace("[ScanOp] Callee: {}, input: {} - {} -> {}",
                           callee.name(),
                           i,
                           kv.first,
                           scopedTensorId);

        auto info = kv.second;

        auto m = i - numVariables;
        if (m >= 0 && m < numScanInputs) {
          // M scan inputs, adjust & drop scan axis
          auto shape       = info.shape();
          scanInputAxes[m] = (scanInputAxes[m] + shape.size()) % shape.size();
          // This squeezing behaviour may not be specified in the standard,
          // but some TF2ONNX models seem to depend on it
          if (shape.size() > 1) {
            shape.erase(shape.begin() + scanInputAxes[m]);
          } else {
            shape[scanInputAxes[m]] = 1;
          }
          info.set(info.dataType(), shape, info.metaShape());
        }
        logging::op::trace("[ScanOp] Adding callee input {} id: {} info: {}",
                           i,
                           scopedTensorId,
                           info);
        calleeGraph.addInput(scopedTensorId, info);
      }

      Op *op = graph.createOp<ScanOp>(info.opid,
                                      info.settings,
                                      calleeGraph,
                                      numScanInputs,
                                      numImplicit,
                                      scanInputAxes,
                                      scanInputDirections,
                                      scanOutputAxes,
                                      scanOutputDirections);

      // Connect explicit inputs
      if (info.hasInputIds()) {
        for (InIndex i = 0; i < info.getInputIds().size(); ++i) {
          auto scopedName =
              graph.getTensors().find(info.getInputIds().at(i), op->getScope());
          op->connectInTensor(i, scopedName);
        }
      }

      // Connect implicit inputs
      for (auto parentScopedImplicitTensorId : parentScopedImplicitTensorIds) {
        op->connectInTensor(op->input->maxIndex() + 1,
                            parentScopedImplicitTensorId);
      }

      // Construct body graph
      calleeGraph.constructFromOnnxGraph(callee);

      // Mark body outputs
      for (TensorId outputId : scanBodyOutputs) {
        TensorId scopedTensorId = addScope(calleeGraph, outputId);
        calleeGraph.markAsOutput(scopedTensorId);
      }

      // Connect outputs
      if (info.hasOutputIds()) {
        for (OutIndex i = 0; i < info.getOutputIds().size(); ++i) {
          op->createAndConnectOutTensor(i, info.getOutputIds().at(i));
        }
      }

      logging::op::trace("[ScanOp] Callee: {}, inputs: {}, outputs: {}",
                         calleeGraph.id.str(),
                         calleeGraph.getInputIds(),
                         calleeGraph.getOutputIds());
      return op;
    },
    true);

} // namespace

} // namespace popart
