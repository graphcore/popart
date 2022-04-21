// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALIASES_MAP_HPP
#define GUARD_NEURALNET_ALIASES_MAP_HPP

#include <map>

#include <popart/aliases.hpp>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>

namespace popart {

/**
 * This class manages a mapping from Graphs to Aliases.
 *
 * NOTE: This is a placeholder class introduced in T40051 and due to be removed
 * again in T39612. Due to the temporary nature of the class it is slightly
 * light on documentation and testing.
 **/
class AliasesMap {
public:
  /**
   * Does not set IR or call update.
   **/
  AliasesMap();

  /**
   * Sets IR and calls update();
   */
  explicit AliasesMap(const Ir *ir);

  /**
   * Sets IR and calls update(graph);
   */
  explicit AliasesMap(const Graph &graph);

  /**
   * Sets IR but does not update.
   **/
  void setIr(const Ir *ir);

  Aliases &getAliases(const GraphId &graphId);
  Aliases &getAliases(Graph &graph);
  const Aliases &getAliases(const GraphId &graphId) const;
  const Aliases &getAliases(Graph &graph) const;

  void clear();
  void update();
  void update(const GraphId &graphId);
  void update(const Graph &graph);
  void update(Op *op);

private:
  const Ir *ir;
  std::map<GraphId, Aliases> aliasesMap;
};

} // namespace popart

#endif
