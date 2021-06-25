// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PREAUTOMATICLOSSSCALING_HPP
#define GUARD_NEURALNET_PREAUTOMATICLOSSSCALING_HPP

#include <cstddef>
#include <string>

#include <popart/graph.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {
/**
 * A transform that annotates tensors in the forward graph, so that their
 * gradients can be tracked in automatic loss scaling.
 *
 * This transform reads a list of user-provided tensor IDs in the forward graph
 * and inserts AutoLossScaleProxyOps after them (see example below). Later in
 * the lowering process, the Autodiff transform will place the corresponding
 * AutoLossScaleProxyGradOps in the backward graph, marking the tensor locations
 * in the graph, for which to track gradients.
 *
 * Example graph before applying the transform:
 * A -- MulOp -- C
 * B -'
 *
 * Example graph after applying the transform with toTrackTensors = ["A", "C"]:
 * A -- AlsProxyOp -- A* -- MulOp -- C -- AlsProxyOp -- C*
 * B ---------------------'
 *
 * It is important to apply the AutomaticLossScale transform after
 * PreAutomaticLossScale and Autodiff to remove all AutoLossScaleProxyOps and
 * AutoLossScaleProxyGradOps.
 */
class PreAutomaticLossScale : public Transform {
public:
  PreAutomaticLossScale() : Transform() {}
  virtual ~PreAutomaticLossScale() override {}

  /**
   * Annotate tensors in the forward graph, so that their gradients can be found
   * and tracked in automatic loss scaling.
   *
   * See class documentation for details.
   *
   * \param graph The graph which to transform.
   * \return true if there was a change to the graph.
   * \return false if there wasn't a change to the graph.
   * \throws error if the user provides an empty list to
   *     automaticLossScalingSettings.toTrackTensors.
   * \throws error if any of the tensor IDs in
   *     automaticLossScalingSettings.toTrackTensors don't exist in the graph.
   */
  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final;

  virtual std::string getName() const final;
};

} // namespace popart

#endif // GUARD_NEURALNET_PREAUTOMATICLOSSSCALING_HPP
