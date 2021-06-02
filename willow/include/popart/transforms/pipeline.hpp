// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PIPELINE_HPP
#define GUARD_NEURALNET_PIPELINE_HPP

#include <popart/op/restore.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

// A helper class for constructing the pipeline on a per-cycle basis
class PipelineInfo {
public:
  PipelineInfo() = default;
  PipelineInfo(int64_t _batchesPerStep,
               int64_t _gradAcclFactor,
               int64_t _maxPipelineStage,
               bool _doTraining,
               bool _doGradAccl);

  bool doTraining;
  bool doGradAccl;

  struct PipelinePhase {
    // [start, end]
    PipelineCycle start, end;
  };

  PipelinePhase fillPhase;

  // The phase between the pipeline being filled and flushed
  PipelinePhase mainPhase;

  PipelinePhase flushPhase;

  bool doStage(PipelineCycle, PipelineStage) const;
};

class Pipeline : public Transform {
public:
  static std::size_t id();

  Pipeline() : Transform() {}
  virtual ~Pipeline() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Pipeline"; }

  static bool inplaceRestoreRequiredForRecompute(Op *);

  static bool inplaceRecomputationConflict(Op *op, InIndex in, OutIndex out);

  static void setFinalFwdStageRecomputation(Graph &graph);

private:
  RestoreOp *addNewRestoreOp(Graph &graph, int64_t stashSize) const;
  RestoreInplaceOp *addNewRestoreInplaceOp(Graph &graph,
                                           int64_t stashSize) const;
};

} // namespace popart

#endif
