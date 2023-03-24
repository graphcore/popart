// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/gather.hpp>
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

GatherOp::GatherOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   int64_t group_size_,
                   const Op::Settings &settings_,
                   const nonstd::optional<float> &available_memory_proportion_,
                   bool zeroOutOfRangeIndices_)
    : Op(_opid, settings_), axis(axis_), group_size(group_size_),
      available_memory_proportion(available_memory_proportion_),
      zeroOutOfRangeIndices__(zeroOutOfRangeIndices_) {}

std::unique_ptr<Op> GatherOp::clone() const {
  return std::make_unique<GatherOp>(*this);
}

std::vector<std::unique_ptr<Op>> GatherOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<GatherGradOp>(*this, axis, group_size));
  return result;
}

int64_t GatherOp::getAxis() const { return axis; }
int64_t GatherOp::getGroupSize() const { return group_size; }

void GatherOp::setup() {
  int64_t axis_min = -static_cast<int64_t>(inShape(dataInIndex()).size());
  int64_t axis_max = inShape(dataInIndex()).size() - 1;
  if (axis_min > axis || axis > axis_max) {
    throw error(
        "GatherOp::setup axis = {} is outside the acceptable range [{}, {}]",
        axis,
        axis_min,
        axis_max);
  }

  // ONNX allows the axis attribute to be negative
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [-m+1, m-1]
  axis += inShape(dataInIndex()).size();       // axis in the range [0, 2m-1]
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [0, m-1]

  if (axis == 0 && group_size > 1) {
    throw error(
        "GatherOp::setup axis = {} indicates the dimension of the group", axis);
  }

  // Replace the axis dimension with the indices shape
  auto data_shape    = inShape(dataInIndex());
  auto indices_shape = inShape(indicesInIndex());
  // Avoiding double addition of a group dimension (with indices and dates).
  if (group_size > 1)
    indices_shape.erase(indices_shape.begin());
  const auto insertion_point = data_shape.erase(data_shape.begin() + axis);

  data_shape.insert(
      insertion_point, indices_shape.begin(), indices_shape.end());

  // Use the computed shape with the data input type
  outInfo(outIndex()) =
      TensorInfo(inInfo(dataInIndex()).dataType(), data_shape);
}

void GatherOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("group_size", group_size);
  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

// A gather on a degenerate dimension with a rank 1 index tensor with a single
// element can be replaced by identity
bool GatherOp::canBeReplacedByIdentity() const {
  return (inShape(dataInIndex())[getAxis()] == 1 &&
          inInfo(indicesInIndex()).rank() == 1 &&
          inInfo(indicesInIndex()).nelms() == 1);
}

GatherGradOp::GatherGradOp(const GatherOp &op,
                           int64_t axis_,
                           int64_t group_size_)
    : Op((op.opid == Onnx::CustomOperators::GroupedGather
              ? Onnx::CustomGradOperators::GroupedGatherGrad
              : Onnx::GradOperators::GatherGrad),
         op.getSettings()),
      axis(axis_), group_size(group_size_),
      fwdDataInfo(op.inInfo(GatherOp::dataInIndex())),
      available_memory_proportion(op.getAvailableMemoryProportion()) {}

std::unique_ptr<Op> GatherGradOp::clone() const {
  return std::make_unique<GatherGradOp>(*this);
}

const std::vector<GradInOutMapper> &GatherGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), GatherOp::outIndex(), GradOpInType::GradOut},
      {indicesInIndex(), GatherOp::indicesInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &GatherGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), GatherOp::dataInIndex()}};

  return outInfo;
}

void GatherGradOp::setup() { outInfo(gradOutIndex()) = fwdDataInfo; }

int64_t GatherGradOp::getAxis() const { return axis; }
int64_t GatherGradOp::getGroupSize() const { return group_size; }

void GatherGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("group_size", group_size);
  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

namespace {

static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL,
                                    DataType::FLOAT8_143,
                                    DataType::FLOAT8_152};
static OpDefinition::DataTypes T1 = {DataType::INT32, DataType::INT64};

static OpDefinition gatherOpDef(
    {OpDefinition::Inputs({{"data", T}, {"indices", T1}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{"axis", {"*"}}, {"group_size", {"*"}}})});

static OpCreator<GatherOp> gatherOpCreator(
    OpDefinitions({{Onnx::Operators::Gather_1, gatherOpDef},
                   {Onnx::Operators::Gather_11, gatherOpDef},
                   {Onnx::CustomOperators::GroupedGather, gatherOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", 0);
      int64_t group_size =
          info.attributes.getAttribute<Attributes::Int>("group_size", 1);

      nonstd::optional<float> available_memory_proportion;

      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        available_memory_proportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new GatherOp(info.opid,
                                              axis,
                                              group_size,
                                              info.settings,
                                              available_memory_proportion));
    },
    true);
} // namespace

} // namespace popart
