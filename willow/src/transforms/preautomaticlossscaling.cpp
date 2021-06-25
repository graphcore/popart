// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <ostream>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/transforms/preautomaticlossscaling.hpp>

namespace popart {

bool PreAutomaticLossScale::apply(Graph &graph) const {
  Ir &ir = graph.getIr();
  Op::Settings gSettings(graph, "op", {});

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);

  const auto &alsSettings = ir.getSessionOptions().automaticLossScalingSettings;

  // Nothing to do, if the user didn't use the toTrackTensors option.
  if (!alsSettings.toTrackTensors.has_value()) {
    logging::transform::debug(
        "[PreAutomaticLossScale] 'automaticLossScalingSettings.toTrackTensors' "
        "is unset. PreAutomaticLossScale leaves the graph unchanged.");
    return false;
  }

  const auto &toTrackTensors = alsSettings.toTrackTensors.value();

  // Verify that toTrackTensors is not empty.
  if (toTrackTensors.size() == 0) {
    throw error("[PreAutomaticLossScale] An empty list was set as the value of "
                "'toTrackTensors'. To disable tensor tracking, please set "
                "'automaticLossScalingSettings.enabled = False'. To track the "
                "default tensors, please leave "
                "'automaticLossScalingSettings.toTrackTensors' unset.");
  }

  // A set to keep track of tensors that have already been annotated. This is
  // used to avoid adding an AutoLossScaleProxyOp to the same tensor more than
  // once.
  std::unordered_set<TensorId> alreadyTracked = {};
  // A flag to track if any of the provided tensors were not found in the graph.
  bool allTensorsExist = true;

  auto idPtr = toTrackTensors.begin();
  for (; idPtr < toTrackTensors.end(); idPtr++) {
    auto id = *idPtr;

    if (alreadyTracked.find(id) != alreadyTracked.end()) {
      logging::transform::warn(
          "[PreAutomaticLossScale] Tensor {} has been added more than once to "
          "the list of 'toTrackTensors'. Please consider removing the "
          "duplicates.",
          id);
    } else {
      if (!ir.containsTensor(id)) {
        allTensorsExist = false;
        break;
      } else {
        alreadyTracked.insert(id);

        auto proxyId = id + "_AlsProxy";
        auto tensor  = ir.getTensor(id);

        auto proxyOp = graph.createOp<AutoLossScaleProxyOp>(
            Onnx::CustomOperators::AutoLossScaleProxy,
            gSettings.copy("AlsProxyOp_" + id));
        proxyOp->connectInTensor(AutoLossScaleProxyOp::getInIndex(), id);
        proxyOp->createAndConnectOutTensor(AutoLossScaleProxyOp::getOutIndex(),
                                           proxyId);
        proxyOp->inheritPlacementAttributes(true, aliasModel);
        proxyOp->setup();

        for (auto &op : tensor->consumers.getOps()) {
          if (op != proxyOp) {
            std::vector<int> indices = op->input->indicesMap().at(tensor);
            for (auto i : indices) {
              op->disconnectInTensor(i, tensor);
              op->connectInTensor(i, proxyId);
            }
          }
        }
      }
    }
  }

  // Alert the user for all tensors that don't exist in the graph and throw.
  if (!allTensorsExist) {
    for (; idPtr < toTrackTensors.end(); idPtr++) {
      auto id = *idPtr;

      if (!ir.containsTensor(id)) {
        logging::transform::err(
            "[PreAutomaticLossScale] Tensor {}, which was added to the "
            "'automaticLossScalingSettings.toTrackTensors' list, does not "
            "exist in the model.",
            id);
      }
    }
    throw error(
        "[PreAutomaticLossScale] Some of the tensors in the "
        "'automaticLossScalingSettings.toTrackTensors' list do not exist in "
        "the model. Please look at the error log.");
  }

  return true;
}

std::size_t PreAutomaticLossScale::getId() const {
  return typeid(PreAutomaticLossScale).hash_code();
}

std::string PreAutomaticLossScale::getName() const {
  return "PreAutomaticLossScale";
}

} // namespace popart
