// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/scatter.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
struct OperatorIdentifier;

ScatterOp::ScatterOp(
    const OperatorIdentifier &_opid,
    int64_t axis_,
    const Op::Settings &settings_,
    const nonstd::optional<float> &available_memory_proportion_)
    : ScatterReduceOp(_opid,
                      axis_,
                      -1,
                      ScatterReduction::None,
                      1 /*group_size_*/,
                      false /*enable_index_broadcast*/,
                      available_memory_proportion_,
                      settings_) {}

std::unique_ptr<Op> ScatterOp::clone() const {
  return std::make_unique<ScatterOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScatterOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  const auto axis = getAxis();
  result.push_back(std::make_unique<ScatterDataGradOp>(*this, axis));
  result.push_back(std::make_unique<ScatterUpdateGradOp>(*this, axis));

  return result;
}

void ScatterOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", getAxis());
  os.appendAttribute(sAvailMemAttribute, getAvailableMemoryProportion());
}

InIndex ScatterOp::srcDataInIndex() const noexcept {
  return ScatterOp::updatesInIndex();
}
InIndex ScatterOp::initialValuesInIndex() const noexcept {
  return ScatterOp::dataInIndex();
}

ScatterDataGradOp::ScatterDataGradOp(const ScatterOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::ScatterDataGrad, op.getSettings()), axis(axis_),
      available_memory_proportion(op.getAvailableMemoryProportion()) {}

std::unique_ptr<Op> ScatterDataGradOp::clone() const {
  return std::make_unique<ScatterDataGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScatterDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), ScatterOp::outIndex(), GradOpInType::GradOut},
      {indicesInIndex(), ScatterOp::indicesInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &ScatterDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), ScatterOp::dataInIndex()}};

  return outInfo;
}

void ScatterDataGradOp::setup() {
  outInfo(gradOutIndex()) = inInfo(gradInIndex());
}

int64_t ScatterDataGradOp::getAxis() const noexcept { return axis; }

void ScatterDataGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

float ScatterDataGradOp::getSubgraphValue() const {
  return getLowSubgraphValue();
}

nonstd::optional<float> ScatterDataGradOp::getAvailableMemoryProportion() const
    noexcept {
  return available_memory_proportion;
}

ScatterUpdateGradOp::ScatterUpdateGradOp(const ScatterOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::ScatterUpdateGrad, op.getSettings()), axis(axis_),
      available_memory_proportion(op.getAvailableMemoryProportion()) {}

std::unique_ptr<Op> ScatterUpdateGradOp::clone() const {
  return std::make_unique<ScatterUpdateGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScatterUpdateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), ScatterOp::outIndex(), GradOpInType::GradOut},
      {indicesInIndex(), ScatterOp::indicesInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &ScatterUpdateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), ScatterOp::updatesInIndex()}};

  return outInfo;
}

void ScatterUpdateGradOp::setup() {
  const auto type         = inInfo(gradInIndex()).dataType();
  outInfo(gradOutIndex()) = TensorInfo(type, inShape(indicesInIndex()));
}

int64_t ScatterUpdateGradOp::getAxis() const noexcept { return axis; }

void ScatterUpdateGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

float ScatterUpdateGradOp::getSubgraphValue() const {
  return getLowSubgraphValue();
}

nonstd::optional<float>
ScatterUpdateGradOp::getAvailableMemoryProportion() const noexcept {
  return available_memory_proportion;
}

namespace {

static OpDefinition::DataTypes T    = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};
static OpDefinition::DataTypes Tind = {DataType::INT32, DataType::INT64};

static OpDefinition scatterOpDef({OpDefinition::Inputs({
                                      {"data", T},
                                      {"indices", Tind},
                                      {"updates", T},
                                  }),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({{"axis", {"*"}}})});

static OpCreator<ScatterOp> ScatterOpCreator(
    OpDefinitions({{Onnx::Operators::Scatter_9, scatterOpDef},
                   {Onnx::Operators::Scatter_11, scatterOpDef},
                   {Onnx::Operators::ScatterElements_11, scatterOpDef}}),
    [](const OpCreatorInfo &info) {
      const int64_t axis =
          info.attributes.getAttribute<Attributes::Int>("axis", 0);

      nonstd::optional<float> available_memory_proportion;

      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        available_memory_proportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new ScatterOp(
          info.opid, axis, info.settings, available_memory_proportion));
    },
    true);
} // namespace

} // namespace popart
