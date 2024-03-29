// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_GRAPHFROMLOSSTOLOSSUPDATER_HPP_
#define POPART_WILLOW_SRC_GRAPHFROMLOSSTOLOSSUPDATER_HPP_

namespace popart {

class Graph;

namespace graphFromLossToLossUpdater {

/**
 * Updates the `PathToLoss Vertex::toLoss` and `PathFromLoss Vertex::fromLoss`
 * attributes of every `Vertex` (`Op` + `Tensor`) in this Graph.
 *
 * The values are propagated forward and backward through the Graph, starting
 * from those Ops for which the attributes are already set.
 */
void propagate(Graph &g);

/**
 * Unset (set to ::Undefined) the `PathToLoss Vertex::toLoss` and `PathFromLoss
 * Vertex::fromLoss` attributes of all vertices in the Graph.
 */
void unsetAll(Graph &g);

} // namespace graphFromLossToLossUpdater
} // namespace popart

#endif // POPART_WILLOW_SRC_GRAPHFROMLOSSTOLOSSUPDATER_HPP_
