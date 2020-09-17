// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGEREMOTE_HPP
#define GUARD_NEURALNET_MERGEREMOTE_HPP

#include <popart/transforms/transform.hpp>

// MergeRemote:
// Transform that merges RemoteLoad & RemoteStore operations
// into a RemoteExchange

namespace popart {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class MergeRemote : public Transform {
public:
  static std::size_t id();

  MergeRemote() : Transform() {}
  virtual ~MergeRemote() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "MergeRemote"; }

private:
  void insertRemoteExchange(Graph &graph, std::vector<Op *> remoteOps) const;
  void conditionallyInsertRemoteExchange(Graph &graph,
                                         std::vector<Op *> remoteOps,
                                         bool phaseMerge,
                                         bool bspMerge) const;
};

} // namespace popart

#endif
