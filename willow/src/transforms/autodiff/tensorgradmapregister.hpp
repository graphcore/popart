// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_TENSORGRADMAPREGISTER_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_TENSORGRADMAPREGISTER_HPP_

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

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_TENSORGRADMAPREGISTER_HPP_
