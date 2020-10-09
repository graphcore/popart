// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>

#include <popart/graph.hpp>
#include <popart/op/cumsum.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace {
void checkAttibs(int64_t exclusive_, int64_t reverse_) {
  if (exclusive_ < 0 || exclusive_ > 1) {
    throw error("CumSum op, 'exclusive' attributes must be 0 or 1.");
  }

  if (reverse_ < 0 || reverse_ > 1) {
    throw error("CumSum op, 'reverse' attributes must be 0 or 1.");
  }
}
} // namespace

CumSumOp::CumSumOp(const OperatorIdentifier &_opid,
                   bool exclusive_,
                   bool reverse_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), exclusive(exclusive_), reverse(reverse_) {}

std::unique_ptr<Op> CumSumOp::clone() const {
  return std::make_unique<CumSumOp>(*this);
}

bool CumSumOp::getExclusive() const { return exclusive; }
bool CumSumOp::getReverse() const { return reverse; }
int64_t CumSumOp::getAxis() const { return axis; }

void CumSumOp::setup() {
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

std::vector<std::unique_ptr<Op>> CumSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(
      std::make_unique<CumSumGradOp>(*this, exclusive, reverse, axis));

  return upops;
}

CumSumGradOp::CumSumGradOp(const CumSumOp &op,
                           bool exclusive_,
                           bool reverse_,
                           int64_t axis_)
    : Op(Onnx::GradOperators::CumSumGrad, op.getSettings()),
      exclusive(exclusive_), reverse(reverse_), axis(axis_),
      fwdOpXInInfo(op.inInfo(CumSumOp::xInIndex())) {}

std::unique_ptr<Op> CumSumGradOp::clone() const {
  return std::make_unique<CumSumGradOp>(*this);
}

const std::vector<GradInOutMapper> &CumSumGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {fwdXInIndex(), CumSumOp::xInIndex(), GradOpInType::In},
      {outGradXInIndex(), CumSumOp::outIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &CumSumGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {outIndex(), CumSumOp::xInIndex()}};

  return outInfo;
}

void CumSumGradOp::setup() { outInfo(outIndex()) = fwdOpXInInfo; }

bool CumSumGradOp::getExclusive() const { return exclusive; }
bool CumSumGradOp::getReverse() const { return reverse; }
int64_t CumSumGradOp::getAxis() const { return axis; }

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
      int64_t exclusiveInt =
          info.attributes.getAttribute<Attributes::Int>("exclusive", 0);
      int64_t reverseInt =
          info.attributes.getAttribute<Attributes::Int>("reverse", 0);

      checkAttibs(exclusiveInt, reverseInt);
      bool exclusive = static_cast<bool>(exclusiveInt);
      bool reverse   = static_cast<bool>(reverseInt);

      return std::unique_ptr<Op>(
          new CumSumOp(info.opid, exclusive, reverse, info.settings));
    },
    true);

} // namespace

} // namespace popart
