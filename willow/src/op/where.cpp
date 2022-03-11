// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <vector>

#include <popart/op/where.hpp>
#include <popart/opmanager.hpp>

namespace popart {

WhereOp::WhereOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> WhereOp::clone() const {
  return std::make_unique<WhereOp>(*this);
}

void WhereOp::setup() {
  outInfo(outIndex()) =
      TensorInfo(inInfo(xInIndex()).dataType(), inShape(conditionInIndex()));
  outInfo(outIndex()) = prettyNpOut(outInfo(outIndex()), inInfo(xInIndex()));
  outInfo(outIndex()) = prettyNpOut(outInfo(outIndex()), inInfo(yInIndex()));
}

std::vector<std::unique_ptr<Op>> WhereOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<WhereXGradOp>(*this));
  upops.emplace_back(std::make_unique<WhereYGradOp>(*this));

  return upops;
}

WhereXGradOp::WhereXGradOp(const WhereOp &op)
    : Op(Onnx::GradOperators::WhereXGrad, op.getSettings()),
      fwdOpXInInfo(op.inInfo(WhereOp::xInIndex())) {}

std::unique_ptr<Op> WhereXGradOp::clone() const {
  return std::make_unique<WhereXGradOp>(*this);
}

const std::vector<GradInOutMapper> &WhereXGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {fwdConditionInIndex(), WhereOp::conditionInIndex(), GradOpInType::In},
      {outGradInIndex(), WhereOp::outIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &WhereXGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{outIndex(), WhereOp::xInIndex()}};

  return outInfo;
}

void WhereXGradOp::setup() { outInfo(outIndex()) = fwdOpXInInfo; }

std::vector<size_t> WhereXGradOp::getFwdInShape() const {
  return fwdOpXInInfo.shape_szt();
}

WhereYGradOp::WhereYGradOp(const WhereOp &op)
    : Op(Onnx::GradOperators::WhereYGrad, op.getSettings()),
      fwdOpYInInfo(op.inInfo(WhereOp::yInIndex())) {}

std::unique_ptr<Op> WhereYGradOp::clone() const {
  return std::make_unique<WhereYGradOp>(*this);
}

const std::vector<GradInOutMapper> &WhereYGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {fwdConditionInIndex(), WhereOp::conditionInIndex(), GradOpInType::In},
      {outGradInIndex(), WhereOp::outIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &WhereYGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{outIndex(), WhereOp::yInIndex()}};

  return outInfo;
}

void WhereYGradOp::setup() { outInfo(outIndex()) = fwdOpYInInfo; }

std::vector<size_t> WhereYGradOp::getFwdInShape() const {
  return fwdOpYInInfo.shape_szt();
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
static OpDefinition::DataTypes TB = {DataType::BOOL};

static OpDefinition
    whereOpDef({OpDefinition::Inputs({{"condition", TB}, {"X", T}, {"Y", T}}),
                OpDefinition::Outputs({{"output", T}}),
                OpDefinition::Attributes({})});

static OpCreator<WhereOp>
    whereOpCreator(OpDefinitions({{Onnx::Operators::Where_9, whereOpDef}}));
} // namespace

} // namespace popart
