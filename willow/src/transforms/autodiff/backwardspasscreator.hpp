// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BACKWARDS_PASS_CREATOR_HPP
#define GUARD_NEURALNET_BACKWARDS_PASS_CREATOR_HPP

#include <popart/names.hpp>

#include <transforms/autodiff/tensorgradmapregister.hpp>

#include <map>
#include <memory>
#include <vector>

namespace popart {

// Forward declaration.
class Op;
class Graph;

class BackwardPassCreator {
public:
  BackwardPassCreator(Graph &fwdGraph_, Graph &bwdGraph_);

private:
  void growGradGraph();
  std::vector<Op *> growGradOps(Op *nonGradOp);
  bool opIsReadyToCreateGradients(Op *);
  void registerBwdOp(Op *fwdOp, Op *bwdOp);
  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &partials);
  TensorId getGradId(const TensorId &);
  void populateGradInInfo(
      const std::map<TensorId, TensorId> &bwdInputIdToFwdTensorId);
  bool hasInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);
  TensorId getInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);

  static void cloneGraph(const Graph &from, Graph &to);
  static void doPrune(Graph &);

  Graph &fwdGraph;
  Graph &bwdGraph;

  // A map of fwd tensors to their corresponding gradient tensors
  std::map<TensorId, TensorId> gradTensorMap;
  TensorGradMapRegister gradRegister;
  std::map<Op *, std::vector<std::unique_ptr<Op>>> gradOpStore;
};

} // namespace popart

#endif