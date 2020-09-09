// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

AdamUpdaterOp::AdamUpdaterOp(AdamMode mode_,
                             OptimizerValue wd,
                             OptimizerValue b1,
                             OptimizerValue b2,
                             OptimizerValue eps,
                             OptimizerValue gs,
                             const Op::Settings &opSettings)
    : Op(Onnx::CustomOperators::AdamUpdater, opSettings), mode(mode_),
      initWd(wd), initB1(b1), initB2(b2), initEps(eps), initGs(gs) {}

void AdamUpdaterOp::setup() {
  outInfo(getUpdaterOutIndex()) = inInfo(getVarInIndex());
}

std::unique_ptr<Op> AdamUpdaterOp::clone() const {
  return std::make_unique<AdamUpdaterOp>(*this);
}

void AdamUpdaterOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initWd.isConst()) {
    os.appendAttribute("const weight decay", initWd.val());
  }

  if (initB1.isConst()) {
    os.appendAttribute("const beta1", initB1.val());
  }

  if (initB2.isConst()) {
    os.appendAttribute("const beta2", initB2.val());
  }

  if (initEps.isConst()) {
    os.appendAttribute("const eps", initEps.val());
  }

  if (initGs.isConst()) {
    os.appendAttribute("const gradient scaling", initGs.val());
  }

  os.appendAttribute("mode", static_cast<int>(mode));
}

view::Regions AdamUpdaterOp::modifies(InIndex index) const {
  if (index == getStepInIndex()) {
    return {view::Region::getFull(inShape(index), view::AccessType::ReadWrite)};
  } else {
    return {view::Region::getEmpty(inRank(index))};
  }
}

ReplicatedTensorShardingIndices
AdamUpdaterOp::getReplicatedTensorShardingIndices() const {
  return {{{AdamUpdaterOp::getVarInIndex(),
            AdamUpdaterOp::getAccl1InIndex(),
            AdamUpdaterOp::getAccl2InIndex()},
           {AdamUpdaterOp::getUpdaterOutIndex()}}};
}

} // namespace popart
