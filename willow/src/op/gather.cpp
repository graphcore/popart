// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <string>
#include <vector>

#include <memory>
#include <popart/op/gather.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

GatherOp::GatherOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_,
                   const nonstd::optional<float> &available_memory_proportion_)
    : Op(_opid, settings_), axis(axis_),
      available_memory_proportion(available_memory_proportion_) {}

std::unique_ptr<Op> GatherOp::clone() const {
  return std::make_unique<GatherOp>(*this);
}

std::vector<std::unique_ptr<Op>> GatherOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<GatherGradOp>(*this, axis));
  return result;
}

int64_t GatherOp::getAxis() const { return axis; }

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

  // Replace the axis dimension with the indices shape
  auto data_shape            = inShape(dataInIndex());
  const auto indices_shape   = inShape(indicesInIndex());
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
  os.appendAttribute("available_memory_proporition",
                     available_memory_proportion);
}

// A gather on a degenerate dimension with a rank 1 index tensor with a single
// element can be replaced by identity
bool GatherOp::canBeReplacedByIdentity() const {
  return (inShape(dataInIndex())[getAxis()] == 1 &&
          inInfo(indicesInIndex()).rank() == 1 &&
          inInfo(indicesInIndex()).nelms() == 1);
}

GatherGradOp::GatherGradOp(const GatherOp &op, int64_t axis_)
    : Op(Onnx::GradOperators::GatherGrad, op.getSettings()), axis(axis_),
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

void GatherGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("available_memory_proporition",
                     available_memory_proportion);
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
                                    DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::INT32, DataType::INT64};

static OpDefinition
    gatherOpDef({OpDefinition::Inputs({{"data", T}, {"indices", T1}}),
                 OpDefinition::Outputs({{"Y", T}}),
                 OpDefinition::Attributes({{"axis", {"*"}}})});

static OpCreator<GatherOp> gatherOpCreator(
    OpDefinitions({{Onnx::Operators::Gather_1, gatherOpDef},
                   {Onnx::Operators::Gather_11, gatherOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", 0);

      nonstd::optional<float> available_memory_proportion;

      if (info.attributes.hasAttribute(sAvailMemAttribute)) {
        available_memory_proportion =
            info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new GatherOp(
          info.opid, axis, info.settings, available_memory_proportion));
    },
    true);
} // namespace

} // namespace popart
