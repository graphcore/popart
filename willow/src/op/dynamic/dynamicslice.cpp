// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

namespace popart {

DynamicSliceOp::DynamicSliceOp(const OperatorIdentifier &_opid,
                               std::vector<int64_t> axes_,
                               std::vector<int64_t> sizes_,
                               bool noOverlap,
                               const Op::Settings &settings_)
    : DynamicSliceBaseOp(_opid, axes_, sizes_, noOverlap, settings_) {}

std::unique_ptr<Op> DynamicSliceOp::clone() const {
  return std::make_unique<DynamicSliceOp>(*this);
}

std::vector<std::unique_ptr<Op>> DynamicSliceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<DynamicSlicePadGradOp>(
      Onnx::CustomGradOperators::DynamicSlicePadGrad,
      getAxes(),
      getSizes(),
      noOverlap,
      settings,
      inInfo(DynamicSliceBaseOp::getInIndex())));
  return upops;
}

// Grad Ops
DynamicSlicePadGradOp::DynamicSlicePadGradOp(const OperatorIdentifier &_opid,
                                             std::vector<int64_t> axes_,
                                             std::vector<int64_t> sizes_,
                                             bool noOverlap_,
                                             const Op::Settings &settings_,
                                             TensorInfo updateInInfo_)
    : DynamicBaseOp(_opid, axes_, sizes_, noOverlap_, settings_),
      updateInInfo(updateInInfo_) {}

void DynamicSlicePadGradOp::setup() { outInfo(getOutIndex()) = updateInInfo; }

std::unique_ptr<Op> DynamicSlicePadGradOp::clone() const {
  return std::make_unique<DynamicSlicePadGradOp>(*this);
}

const std::vector<GradInOutMapper> &
DynamicSlicePadGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), DynamicBaseOp::getOutIndex(), GradOpInType::GRADOUT},
      {getIndexInIndex(), DynamicBaseOp::getIndexInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &DynamicSlicePadGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DynamicSliceOp::getInIndex()}};
  return outInfo;
}

// Creators
static OpDefinition::DataTypes U = {DataType::UINT32};

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    dynamicSliceOpDef({OpDefinition::Inputs({{"X", T}, {"O", U}}),
                       OpDefinition::Outputs({{"Y", T}}),
                       OpDefinition::Attributes({
                           {"axes", {"*"}},
                           {"size", {"*"}},
                           {"noOverlap", {"*"}},
                       })});

static OpCreator<DynamicSliceBaseOp> dynamicSliceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::DynamicSlice_1, dynamicSliceOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes  = attr.getAttribute<Attributes::Ints>("axes");
      std::vector<int64_t> sizes = attr.getAttribute<Attributes::Ints>("sizes");
      bool noOverlap             = attr.hasAttribute("noOverlap") &&
                       attr.getAttribute<Attributes::Int>("noOverlap");
      return std::unique_ptr<DynamicSliceBaseOp>(
          new DynamicSliceOp(_opid, axes, sizes, noOverlap, settings));
    },
    true);

} // namespace popart
