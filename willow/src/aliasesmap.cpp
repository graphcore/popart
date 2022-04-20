// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/aliasesmap.hpp>

#include "popart/aliases.hpp"
#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/graphid.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/region.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace popart {

AliasesMap::AliasesMap() : ir{nullptr}, aliasesMap{} {}

AliasesMap::AliasesMap(const Ir *ir_) : ir{ir_}, aliasesMap{} { update(); }

AliasesMap::AliasesMap(const Graph &graph) : ir{&graph.getIr()}, aliasesMap{} {
  update(graph);
}

void AliasesMap::setIr(const Ir *ir_) { ir = ir_; }

Aliases &AliasesMap::getAliases(const GraphId &graphId) {
  return aliasesMap.at(graphId);
}

Aliases &AliasesMap::getAliases(Graph &graph) {
  return aliasesMap.at(graph.id);
}

const Aliases &AliasesMap::getAliases(const GraphId &graphId) const {
  return aliasesMap.at(graphId);
}

const Aliases &AliasesMap::getAliases(Graph &graph) const {
  return aliasesMap.at(graph.id);
}

void AliasesMap::clear() { aliasesMap.clear(); }

void AliasesMap::update() {
  if (!ir) {
    throw internal_error("[AliasesMap] Ir not set.");
  }
  aliasesMap.clear();
  for (auto &graph : ir->getGraphs()) {
    update(*graph.second.get());
  }
}

void AliasesMap::update(const GraphId &graphId) {
  if (!ir) {
    throw internal_error("[AliasesMap] Ir not set.");
  }
  Graph &graph = ir->getGraph(graphId);
  update(graph);
}

void AliasesMap::update(const Graph &graph) {
  auto &aliases = aliasesMap[graph.id];
  aliases.clearAliases();
  for (auto &op : graph.getOps()) {
    update(op.second.get());
  }
}

void AliasesMap::update(Op *op) {
  if (!ir) {
    throw internal_error("[AliasesMap] Ir not set.");
  }
  logging::trace("[updateAliases] Updating alias for Op {}", op->debugName());

  auto scopedStopwatch =
      ir->timePartitionLogger().scopedStopwatch("Tensors::updateAliases");

  auto &aliases = aliasesMap[op->getGraph().id];

  // for all of the inputs of op, t1 and all output, t2:
  for (auto i1_t1 : op->input->tensorMap()) {
    for (auto o1_t2 : op->output->tensorMap()) {
      InIndex i1 = i1_t1.first;
      Tensor *t1 = i1_t1.second;

      InIndex o1 = o1_t2.first;
      Tensor *t2 = o1_t2.second;

      logging::trace("[updateAliases] In: {}-{} {}, Out: {}-{} {}",
                     i1,
                     t1->id,
                     t1->info.shape(),
                     o1,
                     t2->id,
                     t2->info.shape());

      view::Regions inRegions = op->aliases(i1, o1);

      if (std::all_of(inRegions.begin(), inRegions.end(), [](view::Region &r) {
            return r.isEmpty();
          })) {
        continue;
      }

      auto fwdMap = op->fwdRegMap(i1, o1);
      auto bwdMap = op->bwdRegMap(i1, o1);

      aliases.updateAliases(t1,
                            t2,
                            inRegions,
                            fwdMap,
                            bwdMap,
                            "Fwd Link of " + op->debugName() + " " +
                                std::to_string(i1) + "->" + std::to_string(o1),
                            "Bwd Link of " + op->debugName() + " " +
                                std::to_string(i1) + "->" + std::to_string(o1));
    }
  }
}

} // namespace popart
