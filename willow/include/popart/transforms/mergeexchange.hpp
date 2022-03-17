// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGEEXCHANGE_HPP
#define GUARD_NEURALNET_MERGEEXCHANGE_HPP

#include <popart/names.hpp>
#include <popart/transforms/transform.hpp>

// MergeExchange:
// Transform that merges RemoteLoad/RemoteStore/HostLoad/HostStore operations
// into a MultiExchange

namespace popart {
class ExchangeBaseOp;
class MergeExchange : public Transform {
public:
  static std::size_t id();

  MergeExchange() : Transform() {}
  virtual ~MergeExchange() override {}

  virtual bool apply(Graph &graph) const override;

  std::vector<Op *> applyToOps(Graph &graph,
                               const std::set<OpId> include_ops) const;

  virtual std::size_t getId() const override { return id(); }

  virtual std::string getName() const override { return "MergeExchange"; }

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

#endif
