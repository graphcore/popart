// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/opgradregistry.hpp>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

void OpGradRegistry::insert(Op *nonGrad, int index) {
  auto found = partial.find(nonGrad->id);
  // so far NO gradients for nonGrad are in:
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {}});
  }
  // this should be removed when we're happy the IL (internal logic)
  // is correct:
  if (partial[nonGrad->id].count(index) != 0) {
    throw internal_error("index already present in OpGradRegistry::insert");
  }

  partial[nonGrad->id].insert(index);

  // Check whether an Op is ready to grow gradients based on the set of output
  // indices of `op' for which a gradient is available. Currently, this will
  // just compare the size of the set passed in with number of paths to final
  // loss.
  if (edgesToLoss.at(nonGrad) == partial[nonGrad->id].size()) {
    complete.push_back(nonGrad);
    partial.erase(nonGrad->id);
  }
}

std::vector<Op *> OpGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

void OpGradRegistry::initialize(AutodiffIrInterface &ir) {

  // set all edge counts to zero (we set from scratch in this function)
  for (auto &id_op : ir.getMainGraph().getOps()) {
    Op *op          = id_op.second.get();
    edgesToLoss[op] = 0;
  }

  for (auto &id_op : ir.getMainGraph().getOps()) {
    Op *op = id_op.second.get();

    // For each Op, how many OutIndices lead to loss?
    for (auto index_tensor : op->output->tensorMap()) {
      auto outTensor = index_tensor.second;
      if (outTensor->toLoss == PathToLoss::Yes) {
        ++edgesToLoss[op];
      }
    }
  }
}

} // namespace popart
