// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/lossscaleupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

std::unique_ptr<Op> LossScaleUpdateOp::clone() const {
  return std::make_unique<LossScaleUpdateOp>(*this);
}

void LossScaleUpdateOp::setup() {
  // Verify input shapes:
  // - scalar loss scale and inverse loss scale
  // - 1D, gradient statistics tensors, each of shape [2] (one element for
  //   upper bin counts, one element for lower bin counts)
  auto checkInShape = [&](InIndex idx) {
    if (inShape(idx) != std::vector<int64_t>{}) {
      throw error("LossScaleUpdateOp {}, input {} has unexpected shape. "
                  "Expected shape [], but it is {}",
                  str(),
                  idx,
                  inShape(idx));
    }
  };
  checkInShape(getLossScaleInIndex());
  checkInShape(getInverseLossScaleInIndex());

  if (input->n() < 3) {
    throw error("LossScaleUpdateOp {} must have at least 3 inputs, but it only "
                "has {} inputs.",
                str(),
                input->n());
  }
  for (int i = getFirstStatisticsTensorInIndex(); i < input->n(); i++) {
    if (inShape(i) != std::vector<int64_t>{2}) {
      throw error("LossScaleUpdateOp {}, input {} has unexpected shape. "
                  "Expected shape [2], but it is {}",
                  str(),
                  i,
                  inShape(i));
    }
  }

  outInfo(getUpdatedLossScaleOutIndex()) = inInfo(getLossScaleInIndex());
  outInfo(getUpdatedInverseLossScaleOutIndex()) =
      inInfo(getInverseLossScaleInIndex());
}

view::Regions LossScaleUpdateOp::aliases(InIndex in, OutIndex out) const {
  if ((in == getLossScaleInIndex() && out == getUpdatedLossScaleOutIndex()) ||
      (in == getInverseLossScaleInIndex() &&
       out == getUpdatedInverseLossScaleOutIndex())) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions LossScaleUpdateOp::modifies(InIndex index) const {
  if (index == getLossScaleInIndex()) {
    return aliases(index, getUpdatedLossScaleOutIndex());
  } else if (index == getInverseLossScaleInIndex()) {
    return aliases(index, getUpdatedInverseLossScaleOutIndex());
  } else {
    return {view::Region::getEmpty(inRank(index))};
  }
}

namespace {

static OpDefinition::DataTypes T0 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::UINT32};

// Register the op so that we can add it to an Onnx model via the Builder
// for the purposes of testing
static OpDefinition lossScaleUpdateOpDef(
    {OpDefinition::Inputs({{"loss_scale", T0},
                           {"inverse_loss_scale", T0},
                           {"grad_statistics", T1}}),
     OpDefinition::Outputs({{"update_loss_scale", T0},
                            {"update_inverse_loss_scale", T0}}),
     OpDefinition::Attributes({})});

static OpCreator<LossScaleUpdateOp> lossScaleUpdateOpCreator(OpDefinitions(
    {{Onnx::CustomOperators::LossScaleUpdate, lossScaleUpdateOpDef}}));

} // namespace

} // namespace popart
