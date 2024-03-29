// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ALIASESMAP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ALIASESMAP_HPP_

#include <map>
#include <popart/aliases.hpp>
#include <popart/graphid.hpp>

namespace popart {
class Graph;
class Ir;
class Op;

/**
 * This class manages a mapping from Graphs to Aliases.
 *
 * TODO T39612 Remove this placeholder class introduced in ~T40051~.
 * Due to the temporary nature of the class it is slightly light on
 * documentation and testing.
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

#endif // POPART_WILLOW_INCLUDE_POPART_ALIASESMAP_HPP_
