// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <vector>
#include <popart/op/add.hpp>
#include <popart/opmanager.hpp>
// for `find', we need the algorithm header
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/op/reducesum.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {

// TODO : T6250 : Add support for V6 axis & broadcast attributes

AddOp::AddOp(const OperatorIdentifier &_opid, const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {
  // TODO : Use the attributes in Add-6
}

std::unique_ptr<Op> AddOp::clone() const {
  return std::make_unique<AddOp>(*this);
}

bool AddOp::hasLhsInplaceVariant() const { return true; }

bool AddOp::hasRhsInplaceVariant() const { return true; }

std::unique_ptr<Op> AddOp::getLhsInplaceVariant() const {
  return std::make_unique<AddLhsInplaceOp>(getSettings());
}

std::unique_ptr<Op> AddOp::getRhsInplaceVariant() const {
  return std::make_unique<AddRhsInplaceOp>(getSettings());
}

OperatorIdentifier AddOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::AddLhsInplace;
}

OperatorIdentifier AddOp::getRhsOperatorIdentifier() const {
  return Onnx::CustomOperators::AddRhsInplace;
}

std::unique_ptr<Op> AddLhsInplaceOp::clone() const {
  return std::make_unique<AddLhsInplaceOp>(*this);
}

std::unique_ptr<Op> AddRhsInplaceOp::clone() const {
  return std::make_unique<AddRhsInplaceOp>(*this);
}

AddArg0GradOp::AddArg0GradOp(const Op &op, const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::AddArg0Grad,
                  _axes,
                  false,
                  op.getSettings()),
      forward_op_arg_shape(op.inShape(AddOp::getArg0InIndex())) {}

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

void AddArg0GradOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            forward_op_arg_shape};
}

std::unique_ptr<Op> AddArg0GradOp::clone() const {
  return std::make_unique<AddArg0GradOp>(*this);
}

AddArg1GradOp::AddArg1GradOp(const Op &op, const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::AddArg1Grad,
                  _axes,
                  false,
                  op.getSettings()),
      forward_op_arg_shape(op.inShape(AddOp::getArg1InIndex())) {}

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

void AddArg1GradOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            forward_op_arg_shape};
}

std::unique_ptr<Op> AddArg1GradOp::clone() const {
  return std::make_unique<AddArg1GradOp>(*this);
}

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

static OpCreator<AddLhsInplaceOp> addAddLhsInplaceCreator(
    OpDefinitions({{Onnx::CustomOperators::AddLhsInplace, addOpDef}}));

} // namespace

} // namespace popart
