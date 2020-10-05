// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#include <popart/graph.hpp>
#include <popart/op/cumsum.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

CumSumOp::CumSumOp(const OperatorIdentifier &_opid,
                   int64_t exclusive_,
                   int64_t reverse_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), exclusive(exclusive_), reverse(reverse_) {}

std::unique_ptr<Op> CumSumOp::clone() const {
  return std::make_unique<CumSumOp>(*this);
}

int64_t CumSumOp::getExclusive() const { return exclusive; }
int64_t CumSumOp::getReverse() const { return reverse; }
int64_t CumSumOp::getAxis() const { return axis; }

void CumSumOp::setup() {

  if (exclusive < 0 || exclusive > 1) {
    throw error("CumSum op, 'exclusive' attributes must be 0 or 1.");
  }

  if (reverse < 0 || reverse > 1) {
    throw error("CumSum op, 'reverse' attributes must be 0 or 1.");
  }

  if (inInfo(axisInIndex()).nelms() != 1) {
    throw error("CumSum op, 'axis' tensor must have one element.");
  }

  Tensor *axisTensor = getGraph().getTensors().get(inId(axisInIndex()));
  // Check the input tensor has data.
  if (!axisTensor->hasTensorData()) {
    throw error("For CumSum op {}, the axis tensor {} does not have data.",
                debugName(),
                axisTensor->id);
  }

  // Read axis value from input axis tensor.
  std::vector<int64_t> _axis;
  getInTensorData(
      inId(axisInIndex()), _axis, {DataType::INT32, DataType::INT64});
  axis = _axis[0];

  outInfo(outIndex()) = inInfo(xInIndex());
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT};

static OpDefinition::DataTypes T2 = {DataType::INT32, DataType::INT64};

static OpDefinition cumSumOpDef(
    {OpDefinition::Inputs({{"x", T}, {"axis", T2, true}}),
     OpDefinition::Outputs({{"y", T}}),
     OpDefinition::Attributes({{"exclusive", {"*"}}, {"reverse", {"*"}}})});

static OpCreator<CumSumOp> cumSumOpCreator(
    OpDefinitions({{Onnx::Operators::CumSum_11, cumSumOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t exclusive =
          info.attributes.getAttribute<Attributes::Int>("exclusive", 0);
      int64_t reverse =
          info.attributes.getAttribute<Attributes::Int>("reverse", 0);

      return std::unique_ptr<Op>(
          new CumSumOp(info.opid, exclusive, reverse, info.settings));
    },
    true);

} // namespace

} // namespace popart
