// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <subgraph/wrappedop.hpp>
#include <utility>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/placeholder.hpp>

#include "popart/analysis/replicaequal/replicaequalanalysis.hpp"
#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/subgraph/subgraphnames.hpp"
#include "popart/vendored/any.hpp" // IWYU pragma: keep

using namespace popart;

namespace fwtools {
namespace subgraph {

WrappedOp::WrappedOp(const Op *op_, const ReplicaEqualAnalysis &reAnalysis_)
    : op{op_}, reAnalysis{reAnalysis_}, wrapMap{} {}

std::string WrappedOp::getSubgraphEquivId() {
  auto reAttrs = reAnalysis.get().getOpAttrs(op);
  return op->getSubgraphEquivId(reAttrs);
}

void WrappedOp::setWrapMap(
    const std::map<popart::Op *, WrappedOp *> &wrapMap_) {
  wrapMap = wrapMap_;
}

std::map<fwtools::subgraph::InIndex,
         std::tuple<WrappedOp *, fwtools::subgraph::OutIndex, std::string>>
WrappedOp::getSubgraphInputs() const {
  std::map<fwtools::subgraph::InIndex,
           std::tuple<WrappedOp *, fwtools::subgraph::OutIndex, std::string>>
      res;

  using InIndex  = fwtools::subgraph::OutIndex;
  using OutIndex = fwtools::subgraph::OutIndex;

  using OpTup        = std::tuple<Op *, OutIndex, std::string>;
  using WrappedOpTup = std::tuple<WrappedOp *, OutIndex, std::string>;

  using OpMap        = std::map<InIndex, OpTup>;
  using WrappedOpMap = std::map<InIndex, WrappedOpTup>;

  // Function to map std::tuple<Op*, ...> to std::tuple<WrappedOp*, ...>
  auto opTupToWrappedOpSet = [&](const OpTup &opTup) -> WrappedOpTup {
    return {getWrappedOp(std::get<0>(opTup)),
            std::get<1>(opTup),
            std::get<2>(opTup)};
  };

  // Function to map std::map<InIndex, std::tuple<Op*, ..>> to
  // std::map<InIndex, std::tuple<WrappedOp*, ..>>.
  auto opMapToWrappedOpMap = [&](const OpMap &opMap) -> WrappedOpMap {
    WrappedOpMap wrappedOpMap;
    std::transform(opMap.begin(),
                   opMap.end(),
                   std::inserter(wrappedOpMap, wrappedOpMap.end()),
                   [&](const auto &opEntry) {
                     return std::pair<InIndex, WrappedOpTup>{
                         opEntry.first, opTupToWrappedOpSet(opEntry.second)};
                   });
    return wrappedOpMap;
  };

  return opMapToWrappedOpMap(op->getSubgraphInputs());
}

std::map<fwtools::subgraph::OutIndex, std::set<WrappedOp *>>
WrappedOp::getSubgraphOutputs() const {

  using OutIndex = fwtools::subgraph::OutIndex;

  using WrappedOpSet = std::set<WrappedOp *>;

  using OpMap        = std::map<OutIndex, OpSet>;
  using WrappedOpMap = std::map<OutIndex, WrappedOpSet>;

  // Function to map Op* to WrappedOp*
  auto opToWrappedOp = [&](Op *op) -> WrappedOp * { return getWrappedOp(op); };

  // Function to map OpSet to std::set<WrappedOp*>
  auto opSetToWrappedOpSet = [&](const OpSet &ops) -> WrappedOpSet {
    WrappedOpSet resOps;
    std::transform(ops.begin(),
                   ops.end(),
                   std::inserter(resOps, resOps.end()),
                   opToWrappedOp);
    return resOps;
  };

  // Function to map std::map<OutIndex, OpSet> to
  // std::map<OutIndex, std::set<WrappedOp*>>.
  auto opMapToWrappedOpMap = [&](const OpMap &opMap) -> WrappedOpMap {
    WrappedOpMap wrappedOpMap;
    std::transform(opMap.begin(),
                   opMap.end(),
                   std::inserter(wrappedOpMap, wrappedOpMap.end()),
                   [&](const auto &opEntry) {
                     return std::pair<OutIndex, WrappedOpSet>{
                         opEntry.first, opSetToWrappedOpSet(opEntry.second)};
                   });
    return wrappedOpMap;
  };

  return opMapToWrappedOpMap(op->getSubgraphOutputs());
}

float WrappedOp::getSubgraphValue() const { return op->getSubgraphValue(); }

const popart::Op *WrappedOp::unwrap() const { return op; }

const popart::ReplicaEqualAnalysis &WrappedOp::getReplicaEqualAnalysis() const {
  return reAnalysis.get();
}

WrappedOp *WrappedOp::getWrappedOp(Op *op) const {
  if (op == nullptr) {
    // Op::getSubgraphInputs can return null pointers.
    return nullptr;
  }
  if (op->isConvertibleTo<PlaceholderOp>()) {
    //
  }

  auto findOpIt = wrapMap.find(op);
  if (findOpIt != wrapMap.end()) {
    return findOpIt->second;
  } else {
    throw internal_error("[WrappedOp] Unexpectectedly unable to find Op {} in "
                         "wrapMap",
                         op->debugName());
  }
}

WrappedOpSched toWrappedOpSched(Ir &ir,
                                const ReplicaEqualAnalysis &reAnalysis,
                                const std::vector<Op *> &sched) {

  WrappedOpSched result;
  result.rawPtrs.reserve(sched.size());
  result.sharedPtrs.reserve(sched.size());

  // Populate shared pointers first.
  std::transform(
      sched.begin(),
      sched.end(),
      std::back_inserter(result.sharedPtrs),
      [&](Op *op) { return std::make_shared<WrappedOp>(op, reAnalysis); });

  // Populate shared pointer for placeholder.
  auto placeholder = &ir.getSubgraphAnchorPlaceholder();
  result.sharedPlaceholderPtr =
      std::make_shared<WrappedOp>(placeholder, reAnalysis);

  // Then raw pointers based on shared pointers.
  std::transform(
      result.sharedPtrs.begin(),
      result.sharedPtrs.end(),
      std::back_inserter(result.rawPtrs),
      [&](std::shared_ptr<WrappedOp> &wrappedOp) { return wrappedOp.get(); });

  // Populate raw pointer for placeholder.
  result.rawPlaceholderPtr = result.sharedPlaceholderPtr.get();

  // Work out a mapping from Op* to WrappedOp*.
  std::map<Op *, WrappedOp *> wrapMap;
  for (size_t i = 0; i < sched.size(); ++i) {
    wrapMap[sched[i]] = result.rawPtrs[i];
  }

  // Add placeholder mapping.
  wrapMap[placeholder] = result.rawPlaceholderPtr;

  // Set the wrapMap for every WrappedOp.
  for (auto wrappedOp : result.rawPtrs) {
    wrappedOp->setWrapMap(wrapMap);
  }

  return result;
}

} // namespace subgraph
} // namespace fwtools
