// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGEEXCHANGE_HPP
#define GUARD_NEURALNET_MERGEEXCHANGE_HPP

#include <popart/transforms/transform.hpp>

// MergeExchange:
// Transform that merges RemoteLoad/RemoteStore/HostLoad/HostStore operations
// into a MultiExchange

namespace popart {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class MergeExchange : public Transform {
public:
  static std::size_t id();

  MergeExchange() : Transform() {}
  virtual ~MergeExchange() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MergeExchange"; }

private:
  void insertMultiExchange(Graph &graph,
                           std::vector<ExchangeBaseOp *> exchangeOps) const;
  void
  conditionallyInsertMultiExchange(Graph &graph,
                                   std::vector<ExchangeBaseOp *> exchangeOps,
                                   bool phaseMerge,
                                   bool bspMerge) const;
};

} // namespace popart

#endif
