// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>
#include <popart/graph.hpp>

#include <analysis/replicaequal/replicaequalanalysisimpl.hpp>

namespace popart {

ReplicaEqualAnalysis::ReplicaEqualAnalysis(const Ir &ir)
    : impl{std::make_unique<ReplicaEqualAnalysisImpl>(ir)} {}

ReplicaEqualAnalysis::ReplicaEqualAnalysis(const Ir &ir, AliasModel &aliasModel)
    : impl{std::make_unique<ReplicaEqualAnalysisImpl>(ir, aliasModel)} {}

ReplicaEqualAnalysis::~ReplicaEqualAnalysis() {}

void ReplicaEqualAnalysis::apply() { impl->apply(); }

IsReplicaEqual ReplicaEqualAnalysis::isOpInputEqual(Op *op,
                                                    InIndex inIndex) const {
  return impl->isOpInputEqual(op, inIndex);
}

IsReplicaEqual ReplicaEqualAnalysis::isOpOutputEqual(Op *op,
                                                     OutIndex outIndex) const {
  return impl->isOpOutputEqual(op, outIndex);
}

} // namespace popart
