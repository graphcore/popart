// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORS_GRAD_MAP_REGISTER_HPP
#define GUARD_NEURALNET_TENSORS_GRAD_MAP_REGISTER_HPP

#include <map>
#include <vector>

namespace popart {

// Forward declaration.
class Tensor;

// map of grad Tensor to the list of Tensors that
// must be summed to create the grad Tensor
using GradTensorsPartsMap = std::map<Tensor *, std::vector<Tensor *>>;

class TensorGradMapRegister {
public:
  void insert(Tensor *nonGrad, Tensor *grad);
  GradTensorsPartsMap popComplete();

  GradTensorsPartsMap partial;
  GradTensorsPartsMap complete;
};

} // namespace popart

#endif
