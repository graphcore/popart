// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/hostreducevarupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

GradCopyToHostOp::GradCopyToHostOp(const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::GradCopyToHost, settings_) {}

void GradCopyToHostOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

std::unique_ptr<Op> GradCopyToHostOp::clone() const {
  return std::make_unique<GradCopyToHostOp>(*this);
}

GradCopyFromHostOp::GradCopyFromHostOp(const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::GradCopyFromHost, settings_) {}

void GradCopyFromHostOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

std::unique_ptr<Op> GradCopyFromHostOp::clone() const {
  return std::make_unique<GradCopyFromHostOp>(*this);
}

void GradCopyFromHostOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

HostSGD0VarUpdate::HostSGD0VarUpdate(OptimizerValue slr0,
                                     OptimizerValue wdsf0,
                                     const Op::Settings &settings_)
    : SGD0VarUpdateOpBase(Onnx::CustomOperators::HostSGD0VarUpdate,
                          slr0,
                          wdsf0,
                          settings_) {}

std::unique_ptr<Op> HostSGD0VarUpdate::clone() const {
  return std::make_unique<HostSGD0VarUpdate>(*this);
}

} // namespace popart
