// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP
#define GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// Consider a fragment of the training graph before the explicit recomputation
// transform:
//
// CheckpointOp0
//     |
// RecomputeOp0
//     |
// RecomputeOp1 -.     ...
//     |          \     |
// CheckpointOp1    CheckpointOp1Grad
//    ...              ...
//     |                |
//   Loss --------------
//
// (where CheckpointOp* is an op with
// op->settings.recomputeType == RecomputeType::Checkpoint
// and RecomputeOp* is an op with
// op->settings.recomputeType == RecomputeType::Recompute)
//
// By marking these ops as 'recompute', the output of RecomputeOp1 does not
// need to remain live until the recomputation of CheckpointOp1Grad. In other
// words, the memory used to store this tensor is freed for allocation of other
// tensors as soon as RecomputeOp1's output is read during the computation of
// CheckpointOp1. How does this work in practice?
//
// After the transform, the graph fragment will look like:
//
// CheckpointOp0 -.
//     |           \
// RecomputeOp0   RecomputeOp0Clone
//     |                  |
// RecomputeOp1   RecomputeOp1Clone      ...
//     |                  |               |
// CheckpointOp1           ----- CheckpointOp1Grad
//    ...                                ...
//     |                                  |
//   Loss --------------------------------
//
// The alternative, in the case of implicit recomputation, is to not transform
// the graph at the IR level, and to use these recomputation settings to affect
// the Ir lowering. In this case, the `poplar::program::Sequence`s that
// correspond to the lowered RecomputeOps are added once to the main program as
// scheduled in the forward pass, and then again directly preceding the
// `poplar::program::Sequence` of the CheckpointOp1Grad. See the
// `FindRequiredRecomputes class in irlowering.cpp
class ExplicitRecompute : public Transform {
public:
  static std::size_t id();

  ExplicitRecompute() : Transform() {}
  virtual ~ExplicitRecompute() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "ExplicitRecompute";
  }
};

} // namespace popart

#endif
