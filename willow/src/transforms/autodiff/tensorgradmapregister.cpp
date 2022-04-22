// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/tensorgradmapregister.hpp>
#include <utility>
#include <popart/tensor.hpp>

namespace popart {

void TensorGradMapRegister::insert(Tensor *nonGrad, Tensor *grad) {
  auto found = partial.find(nonGrad);
  if (found == partial.end()) {
    partial.insert({nonGrad, {grad}});
  } else {
    found->second.push_back(grad);
  }

  if (partial.at(nonGrad).size() == nonGrad->consumers.getTotal()) {
    complete.insert({nonGrad, partial.at(nonGrad)});
    partial.erase(nonGrad);
  }
}

GradTensorsPartsMap TensorGradMapRegister::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

} // namespace popart
