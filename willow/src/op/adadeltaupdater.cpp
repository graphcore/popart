// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/adadeltaupdater.hpp>
#include <popart/opserialiser.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

AdaDeltaUpdaterOp::AdaDeltaUpdaterOp(OptimizerValue eps,
                                     const Op::Settings &opSettings)
    : Op(Onnx::CustomOperators::AdaDeltaUpdater, opSettings), initEps(eps) {}

void AdaDeltaUpdaterOp::setup() {
  outInfo(getUpdaterOutIndex()) = inInfo(getGradInIndex());
}

std::unique_ptr<Op> AdaDeltaUpdaterOp::clone() const {
  return std::make_unique<AdaDeltaUpdaterOp>(*this);
}

void AdaDeltaUpdaterOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initEps.isConst()) {
    os.appendAttribute("const eps", initEps.val());
  }
}

ReplicatedTensorShardingIndices
AdaDeltaUpdaterOp::getReplicatedTensorShardingIndices() const {
  return {{{AdaDeltaUpdaterOp::getGradInIndex(),
            AdaDeltaUpdaterOp::getAccl1InIndex(),
            AdaDeltaUpdaterOp::getAccl2InIndex()},
           {AdaDeltaUpdaterOp::getUpdaterOutIndex()}}};
}

} // namespace popart
