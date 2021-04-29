// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP
#define GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP

#include <functional>
#include <list>
#include <map>
#include <set>

#include <popart/logging.hpp>
#include <popart/names.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

// Forward declarations.
class Op;

class OpGradRegistry {
public:
  // Constructor.
  OpGradRegistry(AutodiffIrInterface &ir);

  // register that the output of nonGrad Op at OutIndex index
  // has had its gradient tensor computed
  void insert(Op *nonGrad, int index);
  void fail(Op *nonGrad);

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
  std::reference_wrapper<AutodiffIrInterface> ir;

  // For a non-grad-op, which of its outputs (by index)
  // have had a gradient computed
  std::map<OpId, std::set<int>> partial;
  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::list<Op *> complete;
  // When a required gradient input failed,
  // move the key of partial from partial to failed
  std::list<Op *> failed;

  // Mapping from Op* to the number of outputs it has that lead to the loss.
  std::map<Op *, int> edgesToLoss;
};

} // namespace popart

#endif
