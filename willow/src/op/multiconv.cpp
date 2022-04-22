// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/ir.hpp>
#include <popart/op/multiconv.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/receptive.hpp"
#include "popart/sessionoptions.hpp"

namespace popart {
struct OperatorIdentifier;

MultiConvOp::MultiConvOp(const OperatorIdentifier &_opid,
                         const Settings &settings_,
                         const std::vector<int64_t> &flatStrides_,
                         const std::vector<int64_t> &flatPads_,
                         const std::vector<int64_t> &flatDilations_,
                         const MultiConvOptions &mcOpts_)
    : MultiConvBaseOp(_opid,
                      settings_,
                      flatStrides_,
                      flatPads_,
                      flatDilations_,
                      AutoPad::NOTSET,
                      mcOpts_) {}

std::vector<std::unique_ptr<Op>> MultiConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<MultiConvDataGradOp>(*this));
  upops.emplace_back(std::make_unique<MultiConvWeightsGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> MultiConvOp::clone() const {
  return std::make_unique<MultiConvOp>(*this);
}

void MultiConvOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  MultiConvBaseOp::appendOutlineAttributes(os);

  for (auto key_val : getConvOptions().getGlobalOptions()) {
    os.appendAttribute(key_val.first, key_val.second);
  }
}

MultiConvDataGradOp::MultiConvDataGradOp(const MultiConvOp &op_)
    : MultiConvDataGradBaseOp(op_, Onnx::GradOperators::MultiConvDataGrad) {}

std::unique_ptr<Op> MultiConvDataGradOp::clone() const {
  return std::make_unique<MultiConvDataGradOp>(*this);
}

void MultiConvDataGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  MultiConvDataGradBaseOp::appendOutlineAttributes(os);

  for (auto key_val : getConvOptions().getGlobalOptions()) {
    os.appendAttribute(key_val.first, key_val.second);
  }
}

MultiConvWeightsGradOp::MultiConvWeightsGradOp(const MultiConvOp &op_)
    : MultiConvWeightsGradBaseOp(op_,
                                 Onnx::GradOperators::MultiConvWeightsGrad) {}

std::unique_ptr<Op> MultiConvWeightsGradOp::clone() const {
  return std::make_unique<MultiConvWeightsGradOp>(*this);
}

void MultiConvWeightsGradOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  MultiConvWeightsGradBaseOp::appendOutlineAttributes(os);

  for (auto key_val : getConvOptions().getGlobalOptions()) {
    os.appendAttribute(key_val.first, key_val.second);
  }
}

namespace {
static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    multiconvOpDef({OpDefinition::Inputs({{"inputs", T}}),
                    OpDefinition::Outputs({{"outputs", T}}),
                    OpDefinition::Attributes({{"dilations", {"*"}},
                                              {"pads", {"*"}},
                                              {"strides", {"*"}}})});

static OpCreator<MultiConvOp> multiconvOpCreator(
    OpDefinitions({{Onnx::CustomOperators::MultiConv_1, multiconvOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      // Pass attributes from builder directly into constructor.
      // Design note: we need to provide a default here so that we can
      // create the op inside the Ir without first having set these attributes
      // (which is the case in the MultiConvDataGrad, for example)
      auto attr          = info.attributes;
      auto flatStrides   = attr.getAttribute<Attributes::Ints>("strides", {});
      auto flatPads      = attr.getAttribute<Attributes::Ints>("pads", {});
      auto flatDilations = attr.getAttribute<Attributes::Ints>("dilations", {});

      auto sessOpts =
          info.settings.getIr().getSessionOptions().convolutionOptions;
      auto multiConvOpts = MultiConvOptions(sessOpts, attr);

      return std::unique_ptr<Op>(new MultiConvOp(info.opid,
                                                 info.settings,
                                                 flatStrides,
                                                 flatPads,
                                                 flatDilations,
                                                 multiConvOpts));
    },
    true);

} // namespace

} // namespace popart
