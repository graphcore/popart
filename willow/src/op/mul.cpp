// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/ir.hpp>
#include <popart/op/mul.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

MulOp::MulOp(const OperatorIdentifier &_opid, const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {
  // TODO : Use the attributes in Mul-6
}

std::unique_ptr<Op> MulOp::clone() const {
  return std::make_unique<MulOp>(*this);
}

OperatorIdentifier MulOp::getOpId(const Ir &ir) {
  if (ir.getOpSetVersionFromModel(Domain::ai_onnx) >= 7) {
    return Onnx::Operators::Mul_7;
  } else {
    return Onnx::Operators::Mul_6;
  }
}

std::unique_ptr<Op> MulOp::getLhsInplaceVariant() const {
  return std::make_unique<MulLhsInplaceOp>(getSettings());
}

std::unique_ptr<Op> MulOp::getRhsInplaceVariant() const {
  return std::make_unique<MulRhsInplaceOp>(getSettings());
}

OperatorIdentifier MulOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::MulLhsInplace;
}

OperatorIdentifier MulOp::getRhsOperatorIdentifier() const {
  return Onnx::CustomOperators::MulRhsInplace;
}

bool isFp32Scalar(const TensorInfo &info) {
  return info.dataType() == DataType::FLOAT && info.nelms() == 1;
}

bool isFp16Fp32ScalarMixedPrecision(const TensorInfo &i0,
                                    const TensorInfo &i1) {
  const auto type0 = i0.dataType();
  const auto type1 = i1.dataType();

  return ((isFp32Scalar(i0) || type0 == DataType::FLOAT16) &&
          (isFp32Scalar(i1) || type1 == DataType::FLOAT16) && (type0 != type1));
}

// Numpy broadcasting supports different data types, e.g.:
// >>> np.array([1., 2.]).astype(np.int8) *
//     np.array([3.]).astype(np.float32)
// array([3., 6.], dtype=float32)
//
// but in general mixed precision maths isn't supported when lowering
// to the poplar backend.
//
// The popops::mul and popops::mulInPlace methods are an exception to this is.
// For these calls, fp32-fp16 mixed precision inputs are supported in the case
// that the fp32 tensor is scalar. We therefore use a special type inference
// function for these cases.
DataType getOutputDataType(const TensorInfo &i0,
                           const TensorInfo &i1,
                           const std::string &debugName) {
  if (i0.dataType() == i1.dataType()) {
    return i0.dataType();
  } else if (isFp16Fp32ScalarMixedPrecision(i0, i1)) {
    return DataType::FLOAT16;
  } else {
    throw error(TensorInfo::npOutDataTypeExceptionMessage(i0, i1, debugName));
  }
}

void MulOp::setup() {
  auto out =
      prettyNpOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), false);
  auto outType = getOutputDataType(
      inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), str());
  out.set(outType);
  outInfo(getOutIndex()) = out;
}

std::unique_ptr<Op> MulLhsInplaceOp::clone() const {
  return std::make_unique<MulLhsInplaceOp>(*this);
}

void MulLhsInplaceOp::setup() {
  auto out =
      prettyNpOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), false);
  auto outType = getOutputDataType(
      inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), str());
  out.set(outType);
  outInfo(getOutIndex()) = out;
}

std::unique_ptr<Op> MulRhsInplaceOp::clone() const {
  return std::make_unique<MulRhsInplaceOp>(*this);
}

void MulRhsInplaceOp::setup() {
  auto out =
      prettyNpOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), false);
  auto outType = getOutputDataType(
      inInfo(getArg0InIndex()), inInfo(getArg1InIndex()), str());
  out.set(outType);
  outInfo(getOutIndex()) = out;
}

MulArg0GradOp::MulArg0GradOp(const Op &op_,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::MulArg0Grad,
                                  _reduction_axes,
                                  op_.inInfo(MulOp::getArg0InIndex()),
                                  op_.getSettings()) {}

std::unique_ptr<Op> MulArg0GradOp::clone() const {
  return std::make_unique<MulArg0GradOp>(*this);
}

MulArg1GradOp::MulArg1GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::MulArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(MulOp::getArg1InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> MulArg1GradOp::clone() const {
  return std::make_unique<MulArg1GradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition mulOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<MulOp> mulOpCreator(OpDefinitions(
    {{Onnx::Operators::Mul_6, mulOpDef}, {Onnx::Operators::Mul_7, mulOpDef}}));

} // namespace

} // namespace popart
