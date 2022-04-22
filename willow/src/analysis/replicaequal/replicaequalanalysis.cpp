// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popart/names.hpp"
#include <analysis/replicaequal/replicaequalanalysisimpl.hpp>
#include <map>
#include <memory>
#include <string>
#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>

#include "popart/vendored/any.hpp"

namespace popart {
class AliasModel;
class Ir;
class Op;

ReplicaEqualAnalysis::ReplicaEqualAnalysis(const Ir &ir)
    : impl{std::make_unique<ReplicaEqualAnalysisImpl>(ir)} {}

ReplicaEqualAnalysis::ReplicaEqualAnalysis(const Ir &ir, AliasModel &aliasModel)
    : impl{std::make_unique<ReplicaEqualAnalysisImpl>(ir, aliasModel)} {}

ReplicaEqualAnalysis::~ReplicaEqualAnalysis() {}

void ReplicaEqualAnalysis::apply() { impl->apply(); }

IsReplicaEqual ReplicaEqualAnalysis::isOpInputEqual(const Op *op,
                                                    InIndex inIndex) const {
  return impl->isOpInputEqual(op, inIndex);
}

IsReplicaEqual ReplicaEqualAnalysis::isOpOutputEqual(const Op *op,
                                                     OutIndex outIndex) const {
  return impl->isOpOutputEqual(op, outIndex);
}

std::map<std::string, popart::any>
ReplicaEqualAnalysis::getOpAttrs(const Op *op) const {
  return impl->getOpAttrs(op);
}

} // namespace popart
