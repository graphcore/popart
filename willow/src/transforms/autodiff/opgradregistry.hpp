// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP
#define GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP

#include <functional>
#include <list>
#include <map>
#include <set>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

// Forward declarations.
class Op;
class Graph;

class OpGradRegistry {
public:
  // Constructor.
  OpGradRegistry(Graph &fwdGraph_);

  // register that the output of nonGrad Op at OutIndex index
  // has had its gradient tensor computed
  // When we call insert during initial grad tensors provided
  // we set isProvided to true.
  void insert(Op *nonGrad, int index, bool isProvided = false);
  void fail(Op *nonGrad, int index);

  // Pop one op from the completed pile, if available.
  nonstd::optional<Op *> popComplete();
  // Pop one failed op from the failed pile, if available.
  nonstd::optional<Op *> popFailed();

  // Populate edgesToLoss.
  void initialize();

  // Dump state to logs.
  void logDump(logging::Level level) const;

private:
  // Reference to ir.
  std::reference_wrapper<Graph> fwdGraph;

  // For a non-grad-op, which of its outputs (by index)
  // have had a gradient computed
  // Second element of pair, bool, helps to identify
  // already registered outputs.
  std::map<OpId, std::set<std::pair<int, bool>>> partial;

  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::list<Op *> complete;
  // When a required gradient input failed,
  // move the key of partial from partial to failed
  std::list<Op *> failed;
  // Keep track of Ops that have already been processed so that we don't
  // inadvertently process them twice.
  std::set<OpId> completeOrFailed;

  // Mapping from Op* to the number of outputs it has that lead to the loss.
  std::map<Op *, int> edgesToLoss;
};

} // namespace popart

#endif
