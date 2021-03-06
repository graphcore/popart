// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>

#include <popart/op/scatterreduce.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::string ScatterReduceOp::reductionToString(ScatterReduction reduction) {
  if (reduction == ScatterReduction::Sum) {
    return "sum";
  }

  throw internal_error("Unknown ScatterReduction");
}

ScatterReduction
ScatterReduceOp::reductionFromString(const std::string &reduction) {
  auto lower_arg = boost::algorithm::to_lower_copy(reduction);
  if (lower_arg == "sum") {
    return ScatterReduction::Sum;
  }

  throw internal_error("Unknown ScatterReduction");
}

ScatterReduceOp::ScatterReduceOp(const OperatorIdentifier &_opid,
                                 int64_t axis_,
                                 int64_t axis_size_,
                                 ScatterReduction reduction_,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_), backward_shape(), axis(axis_),
      axis_size(axis_size_), reduction(reduction_) {}

std::unique_ptr<Op> ScatterReduceOp::clone() const {
  return std::make_unique<ScatterReduceOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScatterReduceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ScatterReduceGradOp>(*this));
  return result;
}

void ScatterReduceOp::setup() {
  auto dataShape   = inShape(dataInIndex());
  int64_t dataRank = static_cast<int64_t>(dataShape.size());

  if (-dataRank > axis || axis > dataRank - 1) {
    throw error("ScatterReduceOp::setup axis = {} is outside the acceptable "
                "range [{}, {}]",
                axis,
                -dataRank,
                dataRank - 1);
  }

  // Canonicalise negative axis input
  if (axis < 0) {
    axis += dataRank;
  }

  // indices must have the same shape as the input data
  auto indicesShape = inShape(indicesInIndex());

  if (indicesShape != dataShape) {
    throw error("ScatterReduceOp::setup Mismatch indices and data shape");
  }
  // Output has the same data type as the input and has the same size as the
  // input, except in the axis where the scatter reduction is applied.
  auto dataInfo     = inInfo(dataInIndex());
  auto outputInfo   = dataInfo;
  auto outputShape  = outputInfo.shape();
  outputShape[axis] = axis_size;
  outputInfo.set(outputInfo.dataType(), outputShape);
  outInfo(outIndex()) = outputInfo;
  backward_shape      = dataInfo.shape();
}

void ScatterReduceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("reduction", reductionToString(reduction));
  os.appendAttribute("backward_shape", backward_shape);
}

ScatterReduceGradOp::ScatterReduceGradOp(const ScatterReduceOp &op)
    : Op(Onnx::CustomGradOperators::ScatterReduceGradOp, op.getSettings()),
      backward_shape(op.getBackwardShape()), axis(op.getAxis()),
      reduction(op.getReduction()) {}

std::unique_ptr<Op> ScatterReduceGradOp::clone() const {
  return std::make_unique<ScatterReduceGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScatterReduceGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {gradInIndex(), ScatterReduceOp::outIndex(), GradOpInType::GradOut},
      {indicesInIndex(), ScatterReduceOp::indicesInIndex(), GradOpInType::In}};

  return inInfo;
}

const std::map<int, int> &ScatterReduceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {gradOutIndex(), ScatterReduceOp::dataInIndex()}};

  return outInfo;
}

void ScatterReduceGradOp::setup() {
  const auto type         = inInfo(gradInIndex()).dataType();
  outInfo(gradOutIndex()) = TensorInfo(type, backward_shape);
}

void ScatterReduceGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("reduction",
                     ScatterReduceOp::reductionToString(reduction));
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
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

static OpDefinition::DataTypes Tind = {DataType::UINT8,
                                       DataType::UINT16,
                                       DataType::UINT32,
                                       DataType::UINT64,
                                       DataType::INT8,
                                       DataType::INT16,
                                       DataType::INT32,
                                       DataType::INT64};

static OpDefinition
    scatterReduceOpDef({OpDefinition::Inputs({
                            {"data", T},
                            {"indices", Tind},
                        }),
                        OpDefinition::Outputs({{"output", T}}),
                        OpDefinition::Attributes({{"axis", {"*"}},
                                                  {"axis_shape", {"*"}},
                                                  {"reduction", {"*"}}})});

static OpCreator<ScatterReduceOp> ScatterReduceOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::ScatterReduce, scatterReduceOpDef},
    }),
    [](const OpCreatorInfo &info) {
      int64_t axis_size =
          info.attributes.getAttribute<Attributes::Int>("axis_size");

      if (axis_size < 1) {
        throw error("ScatterReduceOp axis_size = {} is not valid: must be > 0",
                    axis_size);
      }

      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", -1);

      auto reduction =
          info.attributes.getAttribute<Attributes::String>("reduction", "sum");

      return std::unique_ptr<Op>(
          new ScatterReduceOp(info.opid,
                              axis,
                              axis_size,
                              ScatterReduceOp::reductionFromString(reduction),
                              info.settings));
    },
    true);
} // namespace

} // namespace popart
