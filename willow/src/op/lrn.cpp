// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/lrn.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

LRNOp::LRNOp(const OperatorIdentifier &_opid,
             float _alpha,
             float _beta,
             float _bias,
             int64_t _size,
             const Op::Settings &settings_)
    : Op(_opid, settings_), alpha(_alpha), beta(_beta), bias(_bias),
      size(_size) {}

std::unique_ptr<Op> LRNOp::clone() const {
  return std::make_unique<LRNOp>(*this);
}

std::vector<std::unique_ptr<Op>> LRNOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<LRNGradOp>(*this));
  return result;
}

void LRNOp::setup() {
  if (size < 1) {
    throw error("LRN requires size to be >= 1. size is {}", size);
  }

  const auto input_shape = inShape(getInIndex());
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), input_shape};
}

void LRNOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
  os.appendAttribute("bias", bias);
  os.appendAttribute("size", size);
}

LRNGradOp::LRNGradOp(const LRNOp &op_)
    : Op(Onnx::GradOperators::LRNGrad, op_.getSettings()),
      alpha(op_.getAlpha()), beta(op_.getBeta()), bias(op_.getBias()),
      size(op_.getSize()) {}

std::unique_ptr<Op> LRNGradOp::clone() const {
  return std::make_unique<LRNGradOp>(*this);
}

const std::vector<GradInOutMapper> &LRNGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), LRNOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), LRNOp::getInIndex(), GradOpInType::In}};
  return inInfo;
}

void LRNGradOp::setup() {
  const auto input_shape = inShape(getInIndex());
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), input_shape};
}

const std::map<int, int> &LRNGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), LRNOp::getInIndex()}};
  return outInfo;
}

void LRNGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
  os.appendAttribute("bias", bias);
  os.appendAttribute("size", size);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition lrnOpDef({OpDefinition::Inputs({{"X", T}}),
                              OpDefinition::Outputs({{"Y", T}}),
                              OpDefinition::Attributes({
                                  {"alpha", {"*"}},
                                  {"beta", {"*"}},
                                  {"bias", {"*"}},
                                  {"size", {"*"}},
                              })});

static OpCreator<LRNOp> lrnOpCreator(
    OpDefinitions({
        {Onnx::Operators::LRN_1, lrnOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float alpha =
          info.attributes.getAttribute<Attributes::Float>("alpha", 1e-4f);
      float beta =
          info.attributes.getAttribute<Attributes::Float>("beta", 0.75f);
      float bias =
          info.attributes.getAttribute<Attributes::Float>("bias", 1.0f);
      int64_t size = info.attributes.getAttribute<Attributes::Int>("size");

      return std::unique_ptr<Op>(
          new LRNOp(info.opid, alpha, beta, bias, size, info.settings));
    },
    true);

} // namespace

} // namespace popart
