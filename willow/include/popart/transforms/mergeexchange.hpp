// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGEEXCHANGE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGEEXCHANGE_HPP_

#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/names.hpp>
#include <popart/transforms/transform.hpp>

// MergeExchange:
// Transform that merges RemoteLoad/RemoteStore/HostLoad/HostStore operations
// into a MultiExchange

namespace popart {
class ExchangeBaseOp;
class Graph;
class Op;

class MergeExchange : public Transform {
public:
  static std::size_t id();

  MergeExchange() : Transform() {}
  ~MergeExchange() override {}

  bool apply(Graph &graph) const override;

  std::vector<Op *> applyToOps(Graph &graph,
                               const std::set<OpId> include_ops) const;

  std::size_t getId() const override { return id(); }

  std::string getName() const override { return "MergeExchange"; }

private:
  Op *insertMultiExchange(
      Graph &graph,
      std::vector<std::pair<int, ExchangeBaseOp *>> exchangeOps) const;
  Op *conditionallyInsertMultiExchange(
      Graph &graph,
      std::vector<std::pair<int, ExchangeBaseOp *>> exchangeOps,
      const OpsBeforeKey &keys) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGEEXCHANGE_HPP_
