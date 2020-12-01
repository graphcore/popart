// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGELOOPS_HPP
#define GUARD_NEURALNET_MERGELOOPS_HPP

#include <popart/op/loop.hpp>
#include <popart/transforms/transform.hpp>

// MergeLoops:
// Transform that merges multiple compatible loops together
namespace popart {

class MergeLoops : public Transform {
public:
  static std::size_t id();

  MergeLoops() : Transform() {}
  virtual ~MergeLoops() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MergeLoops"; }

private:
  bool canMerge(const LoopOp *const, const LoopOp *const) const;
  bool canMerge(const std::vector<LoopOp *>, const LoopOp *const) const;
  void merge(const std::vector<LoopOp *>) const;

  // Shortcuts paths such as:
  // Op -> Reshape -> DynamicUpdate -> DynamicSlice -> Reshape -> Op
  //       ^^^^^^^                   +                 ^^^^^^^ = Identity
  //                  ^^^^^^^^^^^^^  + ^^^^^^^^^^^^            = Identity
  // to:
  // Op -.-> Reshape -> DynamicUpdate -> DynamicSlice -> Reshape
  //      \
  //       '-> Op
  void shortcutPaths(LoopOp *) const;

  // Removes paths such as:
  // In -> DynamicUpdate -> Out
  // where:
  // 1.) The LoopOp output has no consumer
  // 2.) The path has no internal consumers in the Loop body
  ///    (e.g. because removed by shortcutPaths)
  void prunePaths(LoopOp *) const;

  // Checks paths such as:
  //          Const
  //          |   |
  // In0 -> Add --|----> Out0 (Loop0)
  // In1 -------> Add -> Out1 (Loop1)
  // where the explicit input/output pair is guaranteed to be identical,
  // or the input is implicit, and internal Loop body consumers can
  // therefore be rewired to:
  //          Const
  //          |
  // In0 -> Add --------> Out0 (Loop0)
  bool checkIdenticalPaths(LoopOp *,
                           LoopOp *,
                           InIndex,
                           InIndex,
                           std::map<TensorId, TensorId>) const;
};

} // namespace popart

#endif
