// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP
#define GUARD_NEURALNET_OP_GRAD_REGISTRY_HPP

#include <map>
#include <set>
#include <vector>

#include <popart/names.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>

namespace popart {

// Forward declarations.
class Op;

class OpGradRegistry {
public:
  // register that the output of nonGrad Op at OutIndex index
  // has had its gradient tensor computed
  void insert(Op *nonGrad, int index);
  std::vector<Op *> popComplete();

  // Populate edgesToLoss.
  void initialize(AutodiffIrInterface &ir);

private:
  // For a non-grad-op, which of its outputs (by index)
  // have had a gradient computed
  std::map<OpId, std::set<int>> partial;
  // When all required gradient inputs are in,
  // move the key of partial from partial to complete
  std::vector<Op *> complete;

  // Mapping from Op* to the number of outputs it has that lead to the loss.
  std::map<Op *, int> edgesToLoss;
};

} // namespace popart

#endif
