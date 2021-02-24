// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/scan.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

namespace popart {

ScanOp::ScanOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_,
               int numScanInputs_,
               std::vector<int64_t> scanInputAxes_,
               std::vector<int64_t> scanInputDirections_,
               std::vector<int64_t> scanOutputAxes_,
               std::vector<int64_t> scanOutputDirections_,
               std::vector<std::pair<TensorId, TensorInfo>> opInputs_,
               std::vector<TensorId> implicitTensors_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(-1),
      numImplicitInputs(implicitTensors_.size()), numScanInputs(numScanInputs_),
      scanInputAxes(scanInputAxes_), scanInputDirections(scanInputDirections_),
      scanOutputAxes(scanOutputAxes_),
      scanOutputDirections(scanOutputDirections_) {
  for (int i = 0; i < opInputs_.size(); ++i) {
    TensorId inId = opInputs_.at(i).first;
    if (std::find(implicitTensors_.begin(), implicitTensors_.end(), inId) !=
        implicitTensors_.end()) {
      connectInTensor(i, opInputs_.at(i).first);
    }
  }
}

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
  // Reset trip count, because with ScanOp, the trip count will depend
  // on the length of the scan input tensors at the scan input axes.
  // The new trip count value will be determined when the inputs to the ScanOp
  // are inspected
  tripCountValue = -1;

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

  // Scan inputs
  scanInputAxes.resize(M, 0);
  scanInputDirections.resize(M, 0);

  // Scan outputs
  scanOutputAxes.resize(K, 0);
  scanOutputDirections.resize(K, 0);

  // Check inputs, calculate the number of iterations
  for (int i = 0; i < M; ++i) {
    auto rank = inInfo(N + i).rank();
    // Adjust input axis
    scanInputAxes[i] = (scanInputAxes[i] + rank) % rank;
    auto iterations  = inInfo(N + i).shape().at(scanInputAxes[i]);
    if (tripCountValue == -1) {
      tripCountValue = iterations;
    } else if (tripCountValue != iterations) {
      throw error("[ScanOp] Number of iterations required by the inputs do not "
                  "match: {} {}",
                  tripCountValue,
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
    auto rank     = tensor->info.rank();
    // Adjust output axis
    scanOutputAxes[k] = (scanOutputAxes[k] + rank) % rank;
    auto info         = tensor->info;
    auto shape        = info.shape();
    shape[scanOutputAxes[k]] *= tripCountValue;
    info.set(info.dataType(), shape, info.metaShape());
    outInfo(N + k) = info;
    logging::op::trace("[ScanOp] Output {} info {}", N + k, outInfo(N + k));
  }
}

void ScanOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
  os.appendAttribute("tripCountValue", tripCountValue);
  os.appendAttribute("numImplicitInputs", numImplicitInputs);
  os.appendAttribute("numScanInputs", numScanInputs);
  os.appendAttribute("scanInputAxes", scanInputAxes);
  os.appendAttribute("scanInputDirections", scanInputDirections);
  os.appendAttribute("scanOutputAxes", scanOutputAxes);
  os.appendAttribute("scanOutputDirections", scanOutputDirections);
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
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
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

      for (int i = 0; i < info.getInputIds().size(); ++i) {
        opInputs.push_back({info.getInputIds().at(i),
                            tensors.get(info.getInputIds().at(i))->info});
      }

      int64_t numVariables = (opInputs.size() - numScanInputs);

      scanInputAxes.resize(numScanInputs, 0);

      auto implicitTensors =
          SubgraphOp::getImplicitTensors(callee, tensors, opInputs);

      logging::op::trace("[ScanOp] Implicit tensors: {}", implicitTensors);

      auto subgraphId =
          callee.name().empty()
              ? parentGraph.getIr().createUniqueSubgraphId({"scan"})
              : callee.name();
      auto &ir          = parentGraph.getIr();
      auto &calleeGraph = ir.createGraph(subgraphId);

      int64_t numImplicit = implicitTensors.size();

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
          scopedTensorId = calleeGraph.addScope(scanBodyInputs.at(i));
        } else {
          // L implicit inputs
          scopedTensorId = calleeGraph.addScope(kv.first);
        }
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

      calleeGraph.constructFromOnnxGraph(callee);

      for (TensorId outputId : scanBodyOutputs) {
        TensorId scopedTensorId = calleeGraph.addScope(outputId);
        calleeGraph.markAsOutput(scopedTensorId);
      }

      logging::op::trace("[ScanOp] ScanOp created.");
      return std::unique_ptr<Op>(new ScanOp(info.opid,
                                            info.settings,
                                            calleeGraph,
                                            numScanInputs,
                                            scanInputAxes,
                                            scanInputDirections,
                                            scanOutputAxes,
                                            scanOutputDirections,
                                            opInputs,
                                            implicitTensors));
    },
    true);

} // namespace

} // namespace popart
