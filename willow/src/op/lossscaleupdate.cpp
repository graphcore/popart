// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <string>
#include <vector>
#include <popart/alias/aliasmodel.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/region.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {

std::unique_ptr<Op> LossScaleUpdateOp::clone() const {
  return std::make_unique<LossScaleUpdateOp>(*this);
}

void LossScaleUpdateOp::setup() {
  // Verify input shapes:
  // - 1D, gradient statistics tensor, each of shape [2] (one element for
  //   upper bin counts, one element for lower bin counts)

  if (inShape(getStatisticsTensorInIndex()) != std::vector<int64_t>{2}) {
    throw error("LossScaleUpdateOp {}, input {} has unexpected shape. "
                "Expected shape [2], but it is {}",
                str(),
                getStatisticsTensorInIndex(),
                inShape(getStatisticsTensorInIndex()));
  }

  Shape outShape({}); // scalar tensor
  outInfo(getUpdatedLossScaleUpdateFactorOutIndex()) = {updateFactorDType,
                                                        outShape};
}

view::Regions LossScaleUpdateOp::aliases(InIndex in, OutIndex out) const {
  if (in == getLossScaleUpdateFactorInIndex() &&
      out == getUpdatedLossScaleUpdateFactorOutIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions LossScaleUpdateOp::modifies(InIndex index) const {
  if (index == getLossScaleUpdateFactorInIndex()) {
    return aliases(index, getUpdatedLossScaleUpdateFactorOutIndex());
  } else {
    return {view::Region::getEmpty(inRank(index))};
  }
}

void LossScaleUpdateOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier(*this, getLossScaleUpdateFactorInIndex());
}

namespace {

static OpDefinition::DataTypes T0 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::UINT32};

// Register the op so that we can add it to an Onnx model via the Builder
// for the purposes of testing
static OpDefinition lossScaleUpdateOpDef(
    {OpDefinition::Inputs({{"loss_scale_update_factor", T0},
                           {"grad_statistics", T1}}),
     OpDefinition::Outputs({{"loss_scale_update_factor_updated", T0}}),
     OpDefinition::Attributes({{"to", {"FLOAT|FLOAT16"}}})});

static OpCreator<LossScaleUpdateOp> lossScaleUpdateOpCreator(
    OpDefinitions({{Onnx::CustomOperators::LossScaleUpdate,
                    lossScaleUpdateOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t i64_updateFactorDType;
      info.attributes.set(i64_updateFactorDType, "updateFactorDType");
      DataType updateFactorDType = onnxutil::getDataType(
          static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
              i64_updateFactorDType));

      return std::make_unique<LossScaleUpdateOp>(
          info.opid, updateFactorDType, info.settings);
    },
    true);

} // namespace

} // namespace popart
