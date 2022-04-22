// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>

#include "popart/bwdgraphinfo.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {

CalledGraphGradOpHelper::CalledGraphGradOpHelper(Op *op_)
    : op{op_}, calledGraphsGradInfo() {}

CalledGraphGradOpHelper::~CalledGraphGradOpHelper() {}

void CalledGraphGradOpHelper::setCalledSubgraphGradInfo(
    const FwdGraphToBwdGraphInfo &info_) {
  calledGraphsGradInfo = nonstd::optional<FwdGraphToBwdGraphInfo>(info_);
}

Graph &CalledGraphGradOpHelper::getBwdGraph(SubgraphIndex subgraphIndex) {

  const auto &bwdGraphInfo = getBwdGraphInfo(subgraphIndex);
  auto &bwdGraph           = op->getIr().getGraph(bwdGraphInfo.bwdGraphId);

  return bwdGraph;
}

const FwdGraphToBwdGraphInfo &
CalledGraphGradOpHelper::getCalledSubgraphGradInfo() const {
  throwExceptionIfInfoUnavailable();
  return *calledGraphsGradInfo;
}

const BwdGraphInfo &
CalledGraphGradOpHelper::getBwdGraphInfo(SubgraphIndex subgraphIndex) {

  throwExceptionIfInfoUnavailable();

  // Get the called graph.
  auto calledGraphs = op->getCalledGraphs();
  if (subgraphIndex >= calledGraphs.size()) {
    throw error("[Autodiff] Encountered invalid subgraph index {} "
                "while trying to produce the grad op for {}",
                subgraphIndex,
                op->str());
  }
  const Graph &calledGraph = *calledGraphs.at(subgraphIndex);

  // Check we got bwd graph grad info for calledGraph specifically.
  auto bwdGraphInfoIt = calledGraphsGradInfo->find(calledGraph.id);
  if (bwdGraphInfoIt == calledGraphsGradInfo->end()) {
    throw error("[Autodiff] While trying to produce the grad op "
                "for {}, grad info for called graph '{}' unexpectedly "
                "unavailable",
                op->str(),
                calledGraph.id);
  }

  // Get the BwdGraphInfo for this calledGraph.
  const auto &bwdGraphInfo = bwdGraphInfoIt->second;
  return bwdGraphInfo;
}

std::vector<GradInOutMapper> CalledGraphGradOpHelper::getCalledGraphGradInInfo(
    SubgraphIndex subgraphIndex,
    const std::function<InIndex(InIndex)> &bwdGraphInToGradOpInIndex) {

  // Get the BwdGraphInfo for this calledGraph.
  const auto &bwdGraphInfo = getBwdGraphInfo(subgraphIndex);
  const auto &bwdGraph     = op->getIr().getGraph(bwdGraphInfo.bwdGraphId);
  const Graph &calledGraph = *op->getCalledGraphs().at(subgraphIndex);

  // Populate info about what the inputs of the grad op are.
  std::vector<GradInOutMapper> gradInInfo;
  gradInInfo.reserve(bwdGraphInfo.expectedInputs.size());

  // We need to generate a mapping of the future grad op's input indices to one
  // of:
  //
  // * An input tensor of calledGraph, indicated by GradOpInType::In and the
  //   corresponding non-grad op's input index.
  // * An out tensor of calledGraph, indicated by GradOpInType::Out and the
  //   corresponding non-grad op's output index.
  // * A gradient of an output tensor of calledGraph, indicated by
  //   GradOpInType::OutGrad and the corresponding non-grad op's output index.
  //
  // The information we have available in bwdGraphInfo.expectedInputs details
  // how inputs of bwdGraph are expected to map to tensors (not necessarily
  // inputs and outputs) of calledGraph and whether it should map to the
  // tensor itself of to it's gradient. We need to figure out which case
  // applies and construct the mapping accordingly.
  //
  // However, note this op's input and output indices need not use use the
  // indices of the graph inputs in the called subgraph, so we need to map
  // them using the public methods available in popart::Op.

  for (InIndex i = 0; i < bwdGraphInfo.expectedInputs.size(); ++i) {
    const auto &expConn = bwdGraphInfo.expectedInputs.at(i);

    // Get the output index of bwdGraph we're talking about.
    InIndex bwdGraphInIndex = i;
    // Map it to an output index for the grad op.
    InIndex gradOpInIndex = bwdGraphInToGradOpInIndex(bwdGraphInIndex);

    if (calledGraph.hasOutputId(expConn.fwdId)) {

      // If the fwdId is a graph output of the called graph, we can map that
      // to a tensor as output by the CallOp in this graph and populate the
      // GradInOutMapper accordingly.

      // Get the output index of calledGraph associated with this input.
      OutIndex calledGraphOutIndex = calledGraph.getOutputIndex(expConn.fwdId);
      // Map it to an output index for this op.
      OutIndex nonGradOpOutIndex =
          op->subgraphOutToOpOutIndex(subgraphIndex, calledGraphOutIndex);

      // Deal with it either being the tensor itself or the gradient.
      switch (expConn.type) {
      case ExpectedConnectionType::Fwd: {
        gradInInfo.push_back(
            {gradOpInIndex, nonGradOpOutIndex, GradOpInType::Out});
        break;
      }
      case ExpectedConnectionType::FwdGrad: {
        gradInInfo.push_back(
            {gradOpInIndex, nonGradOpOutIndex, GradOpInType::GradOut});
        break;
      }
      default: {
        throw error("[Autodiff] Unsupported connection type");
      }
      }

    } else if (expConn.type == ExpectedConnectionType::Fwd &&
               calledGraph.hasInputId(expConn.fwdId)) {

      // If the fwdId is a graph input of the called graph, we can map that
      // to a tensor as input by the CallOp in this graph, but we can only
      // populate this if the required connection is the fwd tensor itself and
      // not it's gradient.

      // Get the input index of calledGraph associated with this input.
      InIndex calledGraphInIndex = calledGraph.getInputIndex(expConn.fwdId);
      // Map it to an input index for this op.
      InIndex nonGradOpInIndex =
          op->subgraphInToOpInIndex(subgraphIndex, calledGraphInIndex);

      // If condition guarantees this is not a gradient tensor.
      gradInInfo.push_back({gradOpInIndex, nonGradOpInIndex, GradOpInType::In});

    } else {

      // Copy of inputs / output ids.
      auto calledGraphInputIds  = calledGraph.getInputIds();
      auto calledGraphOutputIds = calledGraph.getOutputIds();

      if (expConn.type == ExpectedConnectionType::Fwd) {
        // Maybe this need not be fatal, but for now it's true in our
        // implementation and it would be good to find out when that changes.
        throw error("[Autodiff] While trying to produce the grad op "
                    "for {}, unexpectedly found {} has input '{}' associated "
                    "with forward tensor '{}', but this associated tensor is "
                    "neither an input of {} (that is, one of: {}) nor an "
                    "output of {} (that is, one of: {})",
                    op->str(),
                    bwdGraph.getGraphString(),
                    bwdGraph.getInputId(i),
                    expConn.fwdId,
                    calledGraph.getGraphString(),
                    logging::join(calledGraphInputIds.begin(),
                                  calledGraphInputIds.end(),
                                  ", "),
                    calledGraph.getGraphString(),
                    logging::join(calledGraphOutputIds.begin(),
                                  calledGraphOutputIds.end(),
                                  ", "));
      } else {
        // Maybe this need not be fatal, but for now it's true in our
        // implementation and it would be good to find out when that changes.
        throw error("[Autodiff] While trying to produce the grad op "
                    "for {}, unexpectedly found {} has input '{}' associated "
                    "with the gradient of forward tensor '{}', but this "
                    "associated tensor is not an output of {} (that is, one "
                    "of: {})",
                    op->str(),
                    bwdGraph.getGraphString(),
                    bwdGraph.getInputId(i),
                    expConn.fwdId,
                    calledGraph.getGraphString(),
                    logging::join(calledGraphOutputIds.begin(),
                                  calledGraphOutputIds.end(),
                                  ", "));
      }
    }
  }

  return gradInInfo;
}

const std::map<int, int>
CalledGraphGradOpHelper::getCalledGraphGradOutToNonGradIn(
    SubgraphIndex subgraphIndex,
    const std::function<OutIndex(OutIndex)> &bwdGraphOutToGradOpOutIndex) {

  // Get the BwdGraphInfo for this calledGraph.
  const auto &bwdGraphInfo = getBwdGraphInfo(subgraphIndex);
  const auto &bwdGraph     = op->getIr().getGraph(bwdGraphInfo.bwdGraphId);
  const Graph &calledGraph = *op->getCalledGraphs().at(subgraphIndex);

  std::map<int, int> gradOutToNonGradIn;

  // We need to generate a mapping of the future grad op's output indices to
  // this op's input indices, to show for which inputs gradients are created.
  //
  // The information we have available in bwdGraphInfo.expectedOutputs details
  // how outputs of bwdGraph are expected to map to inputs of calledGraph.
  // However, op's input and output indices need not use use the indices of the
  // graph inputs in the called subgraph, so we need to map them using the
  // public methods available in popart::Op.

  for (OutIndex i = 0; i < bwdGraphInfo.expectedOutputs.size(); ++i) {
    const auto &expConn = bwdGraphInfo.expectedOutputs.at(i);

    bool calledGraphHasFwdIdAsInput = calledGraph.hasInputId(expConn.fwdId);
    bool isFwdGrad = expConn.type == ExpectedConnectionType::FwdGrad;

    if (calledGraphHasFwdIdAsInput && isFwdGrad) {

      // Get the output index of bwdGraph we're talking about.
      OutIndex bwdGraphOutIndex = i;
      // Map it to an output index for the grad op.
      OutIndex gradOpOutIndex = bwdGraphOutToGradOpOutIndex(bwdGraphOutIndex);

      // Get the input index of calledGraph associated with this output.
      InIndex calledGraphInIndex = calledGraph.getInputIndex(expConn.fwdId);
      // Map it to an input index for this op.
      InIndex nonGradOpInIndex =
          op->subgraphInToOpInIndex(subgraphIndex, calledGraphInIndex);

      // Store it.
      gradOutToNonGradIn[gradOpOutIndex] = nonGradOpInIndex;

    } else {

      // Copy of inputs / output ids.
      auto calledGraphInputIds = calledGraph.getInputIds();

      if (expConn.type == ExpectedConnectionType::Fwd) {
        // Maybe this need not be fatal, but for now it's true in our
        // implementation and it would be good to find out when that changes.
        throw error("[Autodiff] While trying to produce the grad op "
                    "for {}, unexpectedly found {} has output '{}' associated "
                    "with forward tensor '{}', but was expecting it to be "
                    "associated with the gradient of an input of {} (that is, "
                    "one of: {})",
                    op->str(),
                    bwdGraph.getGraphString(),
                    bwdGraph.getOutputId(i),
                    expConn.fwdId,
                    calledGraph.getGraphString(),
                    logging::join(calledGraphInputIds.begin(),
                                  calledGraphInputIds.end(),
                                  ", "));
      } else {
        // Maybe this need not be fatal, but for now it's true in our
        // implementation and it would be good to find out when that changes.
        throw error("[Autodiff] While trying to produce the grad op "
                    "for {}, unexpectedly found {} has output '{}' associated "
                    "with the gradient of forward tensor '{}', but this "
                    "associated tensor is not an input of {} (that is, one of: "
                    "{})",
                    op->str(),
                    bwdGraph.getGraphString(),
                    bwdGraph.getOutputId(i),
                    expConn.fwdId,
                    calledGraph.getGraphString(),
                    logging::join(calledGraphInputIds.begin(),
                                  calledGraphInputIds.end(),
                                  ", "));
      }
    }
  }

  return gradOutToNonGradIn;
}

void CalledGraphGradOpHelper::throwExceptionIfInfoUnavailable() const {
  // Check we have any bwd graph grad info.
  if (!calledGraphsGradInfo) {
    throw error("[Autodiff] While trying to produce the grad op "
                "for {}, grad info for called graphs unexpectedly not set "
                "(make sure to call setCalledSubgraphGradInfo before calling "
                "getGradOp)",
                op->str());
  }
}

} // namespace popart
