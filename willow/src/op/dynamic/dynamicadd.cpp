// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicadd.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

DynamicAddOp::DynamicAddOp(const OperatorIdentifier &_opid,
                           std::vector<int64_t> axes_,
                           std::vector<int64_t> sizes_,
                           bool noOverlap_,
                           const Op::Settings &settings_,
                           TensorInfo updateInInfo_)
    : DynamicTernaryBaseOp(_opid,
                           axes_,
                           sizes_,
                           noOverlap_,
                           settings_,
                           updateInInfo_) {}

std::unique_ptr<Op> DynamicAddOp::clone() const {
  return std::make_unique<DynamicAddOp>(*this);
}

std::vector<std::unique_ptr<Op>> DynamicAddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<IdentityGradOp>(settings));
  upops.emplace_back(std::make_unique<DynamicUpdateUpdaterGradOp>(
      Onnx::CustomGradOperators::DynamicUpdateUpdaterGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings));
  return upops;
}

std::unique_ptr<Op>
DynamicAddOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::DynamicAddInplace) {
    return std::make_unique<DynamicAddInplaceOp>(*this);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
DynamicAddOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::DynamicAddInplace, 10.f}};
}

DynamicAddInplaceOp::DynamicAddInplaceOp(const DynamicAddOp &dynamicAddOp)
    : DynamicTernaryBaseInplaceOp(Onnx::CustomOperators::DynamicAddInplace,
                                  dynamicAddOp.getAxes(),
                                  dynamicAddOp.getSizes(),
                                  dynamicAddOp.isNotOverlapping(),
                                  dynamicAddOp.getSettings(),
                                  dynamicAddOp.getUpdateTensorInfo()) {}

DynamicAddInplaceOp::DynamicAddInplaceOp(const OperatorIdentifier &_opid,
                                         std::vector<int64_t> axes_,
                                         std::vector<int64_t> sizes_,
                                         bool noOverlap_,
                                         const Op::Settings &settings_,
                                         TensorInfo updateInInfo_)
    : DynamicTernaryBaseInplaceOp(_opid,
                                  axes_,
                                  sizes_,
                                  noOverlap_,
                                  settings_,
                                  updateInInfo_) {}

std::unique_ptr<Op> DynamicAddInplaceOp::clone() const {
  return std::make_unique<DynamicAddInplaceOp>(*this);
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    dynamicAddOpDef({OpDefinition::Inputs({{"Y", T}, {"O", U}, {"X", T}}),
                     OpDefinition::Outputs({{"Z", T}}),
                     OpDefinition::Attributes({
                         {"axes", {"*"}},
                         {"size", {"*"}},
                         {"noOverlap", {"*"}},
                     })});

static OpCreator<DynamicUpdateOp> dynamicAddOpCreator(
    OpDefinitions({{Onnx::CustomOperators::DynamicAdd_1, dynamicAddOpDef}}),
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> axes =
          info.attributes.getAttribute<Attributes::Ints>("axes");
      std::vector<int64_t> sizes =
          info.attributes.getAttribute<Attributes::Ints>("sizes");
      bool noOverlap =
          info.attributes.hasAttribute("noOverlap") &&
          info.attributes.getAttribute<Attributes::Int>("noOverlap");
      return std::unique_ptr<DynamicAddOp>(
          new DynamicAddOp(info.opid, axes, sizes, noOverlap, info.settings));
    },
    true);

} // namespace popart
