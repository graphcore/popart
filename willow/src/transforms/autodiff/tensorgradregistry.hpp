// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSOR_GRAD_REGISTRY_HPP
#define GUARD_NEURALNET_TENSOR_GRAD_REGISTRY_HPP

#include <popart/names.hpp>
#include <popart/tensor.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>

namespace popart {

// The gradient of a tensor is the sum of 1 or several tensors,
// 1 for each of the nodes which consumed it. This class is for
// tracking/counting these as they come in down edges in backwards
// part of the training compute graph.
class TensorGradRegistry {
public:
  using TMap = std::map<TensorId, std::vector<Tensor *>>;
  // Register tensor "edgeGrad" as being a
  // gradient of "nonGrad" w.r.t. loss along a single edge
  void insert(Tensor *nonGrad, Tensor *edgeGrad);

  // Decrease the number of edges expected to be registered
  // for a non-grad tensor.
  void decrementNumberExpectedEdges(Tensor *nonGrad);

  int getNumberExpectedEdges(Tensor *nonGrad);

  // Return the non-gradient tensors which have ALL their
  // required gradients registered, and are thus ready to
  // have their edge gradients summed to
  // obtain the final gradient.
  // Note that this is NOT a const pop member function
  TMap popComplete();

  // stores all non-grad tensors which have some, but not all of
  // their edges already having gradients registered
  TMap partial;
  // stores all non-grad tensors which have had all of their
  // edges provide gradients. When popCompleted() is called,
  // this map is returned,
  TMap complete;

  // Populate edgesToLoss.
  void initialize(AutodiffIrInterface &ir);

private:
  // the number of edges expected to register gradients for a non-grad tensor.
  std::map<TensorId, int> expectedNumEdges;

  void tryMakeComplete(Tensor *nonGrad);

  // Mapping from Tensor* to the number of times it appears on a path to loss.
  std::map<Tensor *, int> edgesToLoss;
};

} // namespace popart

#endif