// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_AUTODIFFIRADAPTER_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_AUTODIFFIRADAPTER_HPP_

#include <functional>
#include <string>
#include <transforms/autodiff/autodiffirinterface.hpp>
#include <vector>

#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {
class Graph;
class Ir;
class Optimizer;
class Tensors;
struct SessionOptions;

/**
 * Concrete class used to fulfill all dependencies for the autodiff classes.
 * It's basically a selective wrapper around popart::Ir.
 */
class AutodiffIrAdapter : public AutodiffIrInterface {
public:
  // Constructor.
  explicit AutodiffIrAdapter(Ir &ir);

  // Get a reference to the main graph.
  Graph &getMainGraph() override;
  // Get schedule order of graphs where parent comes before child.
  std::vector<const Graph *> getGraphSchedule() override;
  // Get schedule order of graphs where parent comes before child.
  std::vector<const Graph *> getGraphSchedule(GraphId root) override;
  // Determine if graph exists.
  bool hasGraph(const GraphId &) const override;
  // Get graph.
  Graph &getGraph(const GraphId &) override;
  // Create a new graph.
  Graph &createGraph(const GraphId &) override;
  // Get the main graph's tensors.
  Tensors &getTensors() override;
  // Get a reference to session options.
  const SessionOptions &getSessionOptions() override;
  // Get optimizer.
  const Optimizer &getOptimizer() override;
  // Get the ID of the final loss tensor.
  TensorId getFinalLossId() override;
  // Get the ID of the op that produces the final loss.
  OpId getFinalLossOpId() override;
  // Get the op set version for a domain.
  int getOpSetVersionFromModel(const std::string &domain) override;
  // Set 'path from loss' flag for all gradient tensors and ops producing them.
  void setMainGraphPathFromLoss() override;
  // Final loss' pipeline stage.
  PipelineStage getFinalLossPipelineStage() override;
  // Maximum pipeline stage
  PipelineStage getMaxPipelineStage() override;
  // Create a new tensor id.
  TensorId createIntermediateTensorId(const TensorId &base_id) override;

private:
  // Reference to ir.
  std::reference_wrapper<Ir> ir;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_AUTODIFFIRADAPTER_HPP_
