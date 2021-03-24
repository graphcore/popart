// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/opgradregistry.hpp>

#include <popart/op.hpp>

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

  // probably just checks that the size of partial is
  // nonGrad->output->n(), but maybe not.
  if (nonGrad->readyToCreateGradients(partial[nonGrad->id])) {
    complete.push_back(nonGrad);
    partial.erase(nonGrad->id);
  }
}

std::vector<Op *> OpGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

} // namespace popart
