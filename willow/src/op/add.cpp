// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <vector>
// for `find', we need the algorithm header
#include <algorithm>
#include <memory>
#include <popart/op/add.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

// TODO : T6250 : Add support for V6 axis & broadcast attributes

AddOp::AddOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {

  // TODO : Use the attributes in Add-6
}

std::unique_ptr<Op> AddOp::clone() const {
  return std::make_unique<AddOp>(*this);
}

std::vector<std::unique_ptr<Op>> AddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = inShape(getArg0InIndex());
  const auto &shape_a1 = inShape(getArg1InIndex());
  const auto &shape_o0 = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<AddArg0GradOp>(
      *this, npReductionAxis(shape_a0, shape_o0)));
  upops.emplace_back(std::make_unique<AddArg1GradOp>(
      *this, npReductionAxis(shape_a1, shape_o0)));

  return upops;
}

bool AddOp::hasLhsInplaceVariant() const { return true; }

bool AddOp::hasRhsInplaceVariant() const { return true; }

std::unique_ptr<Op> AddOp::getLhsInplaceVariant() const {
  return std::make_unique<AddLhsInplaceOp>(*this);
}

std::unique_ptr<Op> AddOp::getRhsInplaceVariant() const {
  return std::make_unique<AddRhsInplaceOp>(*this);
}

OperatorIdentifier AddOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::AddLhsInplace;
}

OperatorIdentifier AddOp::getRhsOperatorIdentifier() const {
  return Onnx::CustomOperators::AddRhsInplace;
}

AddLhsInplaceOp::AddLhsInplaceOp(const AddOp &op)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::AddLhsInplace,
                                    op.getSettings()) {}

AddLhsInplaceOp::AddLhsInplaceOp(const Op::Settings &settings_)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::AddLhsInplace,
                                    settings_) {}

std::unique_ptr<Op> AddLhsInplaceOp::clone() const {
  return std::make_unique<AddLhsInplaceOp>(*this);
}

AddRhsInplaceOp::AddRhsInplaceOp(const AddOp &op)
    : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::AddRhsInplace,
                                    op.getSettings()) {}

AddRhsInplaceOp::AddRhsInplaceOp(const Op::Settings &settings_)
    : ElementWiseBinaryInplaceRhsOp(Onnx::CustomOperators::AddRhsInplace,
                                    settings_) {}

std::unique_ptr<Op> AddRhsInplaceOp::clone() const {
  return std::make_unique<AddRhsInplaceOp>(*this);
}

AddArg0GradOp::AddArg0GradOp(const AddOp &op_,
                             const std::vector<int64_t> &axes_)
    : ReduceSumOp(Onnx::GradOperators::AddArg0Grad,
                  axes_,
                  false,
                  op_.getSettings()),
      forward_op_arg_info(op_.inInfo(AddOp::getArg0InIndex())) {}

const std::map<int, int> &AddArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

void AddArg0GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

AddArg1GradOp::AddArg1GradOp(const AddOp &op_,
                             const std::vector<int64_t> &axes_)
    : ReduceSumOp(Onnx::GradOperators::AddArg1Grad,
                  axes_,
                  false,
                  op_.getSettings()),
      forward_op_arg_info(op_.inInfo(AddOp::getArg1InIndex())) {}

const std::map<int, int> &AddArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg1GradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of add
  // might need to reduce across certain axes of this
  // if numpy broadcasting happened
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

void AddArg1GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition addOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<AddOp> addOpCreator(OpDefinitions(
    {{Onnx::Operators::Add_6, addOpDef}, {Onnx::Operators::Add_7, addOpDef}}));

} // namespace

} // namespace popart
