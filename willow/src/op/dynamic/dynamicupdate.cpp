// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

DynamicUpdateOp::DynamicUpdateOp(const OperatorIdentifier &_opid,
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

std::unique_ptr<Op> DynamicUpdateOp::clone() const {
  return std::make_unique<DynamicUpdateOp>(*this);
}

std::vector<std::unique_ptr<Op>> DynamicUpdateOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(std::make_unique<DynamicUpdateUpdaterGradOp>(
      Onnx::CustomGradOperators::DynamicUpdateUpdaterGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings));

  upops.emplace_back(std::make_unique<DynamicUpdateToUpdateGradOp>(
      Onnx::CustomGradOperators::DynamicUpdateToUpdateGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings));

  return upops;
}

std::unique_ptr<Op> DynamicUpdateOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::DynamicUpdateInplace) {
    return std::make_unique<DynamicUpdateInplaceOp>(*this);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
DynamicUpdateOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::DynamicUpdateInplace, 10.f}};
}

DynamicUpdateInplaceOp::DynamicUpdateInplaceOp(
    const DynamicUpdateOp &dynamicUpdateOp)
    : DynamicTernaryBaseInplaceOp(Onnx::CustomOperators::DynamicUpdateInplace,
                                  dynamicUpdateOp.getAxes(),
                                  dynamicUpdateOp.getSizes(),
                                  dynamicUpdateOp.isNotOverlapping(),
                                  dynamicUpdateOp.getSettings(),
                                  dynamicUpdateOp.getUpdateTensorInfo()) {}

DynamicUpdateInplaceOp::DynamicUpdateInplaceOp(const OperatorIdentifier &_opid,
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

std::unique_ptr<Op> DynamicUpdateInplaceOp::clone() const {
  return std::make_unique<DynamicUpdateInplaceOp>(*this);
}

DynamicUpdateToUpdateGradOp::DynamicUpdateToUpdateGradOp(
    const OperatorIdentifier &_opid,
    std::vector<int64_t> axes_,
    std::vector<int64_t> sizes_,
    bool noOverlap_,
    const Op::Settings &settings_)
    : DynamicBinaryBaseOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

std::unique_ptr<Op> DynamicUpdateToUpdateGradOp::clone() const {
  return std::make_unique<DynamicUpdateToUpdateGradOp>(*this);
}

const std::vector<GradInOutMapper> &
DynamicUpdateToUpdateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getUpdateInIndex(),
       DynamicBinaryBaseOp::getOutIndex(),
       GradOpInType::GradOut},
      {getIndexInIndex(),
       DynamicBinaryBaseOp::getIndexInIndex(),
       GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &
DynamicUpdateToUpdateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DynamicBinaryBaseOp::getUpdateInIndex()}};
  return outInfo;
}

DynamicUpdateUpdaterGradOp::DynamicUpdateUpdaterGradOp(
    const OperatorIdentifier &_opid,
    std::vector<int64_t> axes_,
    std::vector<int64_t> sizes_,
    bool noOverlap_,
    const Op::Settings &settings_)
    : DynamicSliceBaseOp(_opid, axes_, sizes_, noOverlap_, settings_) {}

std::unique_ptr<Op> DynamicUpdateUpdaterGradOp::clone() const {
  return std::make_unique<DynamicUpdateUpdaterGradOp>(*this);
}

const std::vector<GradInOutMapper> &
DynamicUpdateUpdaterGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), DynamicUpdateOp::getOutIndex(), GradOpInType::GradOut},
      {getIndexInIndex(),
       DynamicUpdateOp::getIndexInIndex(),
       GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &
DynamicUpdateUpdaterGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DynamicUpdateOp::getInIndex()}};
  return outInfo;
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    dynamicUpdateOpDef({OpDefinition::Inputs({{"Y", T}, {"O", U}, {"X", T}}),
                        OpDefinition::Outputs({{"Z", T}}),
                        OpDefinition::Attributes({
                            {"axes", {"*"}},
                            {"size", {"*"}},
                            {"noOverlap", {"*"}},
                        })});

static OpCreator<DynamicUpdateOp> dynamicUpdateOpCreator(
    OpDefinitions({{Onnx::CustomOperators::DynamicUpdate_1,
                    dynamicUpdateOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes  = attr.getAttribute<Attributes::Ints>("axes");
      std::vector<int64_t> sizes = attr.getAttribute<Attributes::Ints>("sizes");
      bool noOverlap             = attr.hasAttribute("noOverlap") &&
                       attr.getAttribute<Attributes::Int>("noOverlap");
      return std::unique_ptr<DynamicUpdateOp>(
          new DynamicUpdateOp(_opid, axes, sizes, noOverlap, settings));
    },
    true);

} // namespace popart
