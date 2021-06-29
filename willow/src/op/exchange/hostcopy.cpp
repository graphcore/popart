// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

HostLoadOp::HostLoadOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_,
                       TensorId sid_)
    : ExchangeBaseOp(_opid, settings_), hostStreamTensorId(sid_) {}

std::unique_ptr<Op> HostLoadOp::clone() const {
  return std::make_unique<HostLoadOp>(*this);
}

void HostLoadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hostStreamTensorId", hostStreamTensorId);
}

void HostLoadOp::setup() {
  logging::op::debug("HostLoadOp setup started");
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
}

view::Regions HostLoadOp::modifies(InIndex index) const {
  if (index == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else {
    throw error("Invalid index passed to HostLoadOp::modifies");
  }
}

view::Regions HostLoadOp::aliases(InIndex in, OutIndex) const {
  if (in == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    throw error("Invalid index passed to HostLoadOp::aliases");
  }
}

view::RegMap HostLoadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap HostLoadOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  return Op::bwdRegMap(inIndex, outIndex);
}

ExchangeDescriptor HostLoadOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            hostStreamTensorId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n());
}

HostStoreOp::HostStoreOp(const OperatorIdentifier &_opid,
                         const Op::Settings &settings_,
                         TensorId sid_)
    : ExchangeBaseOp(_opid, settings_), hostStreamTensorId(sid_) {}

std::unique_ptr<Op> HostStoreOp::clone() const {
  return std::make_unique<HostStoreOp>(*this);
}

void HostStoreOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hostStreamTensorId", hostStreamTensorId);
}

void HostStoreOp::setup() { logging::op::debug("HostStoreOp setup started"); }

ExchangeDescriptor HostStoreOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Store,
                            hostStreamTensorId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n());
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition hostLoadOpDef({OpDefinition::Inputs({{"X", T}}),
                                   OpDefinition::Outputs({{"Y", T}}),
                                   OpDefinition::Attributes({})});

static OpCreator<HostLoadOp> hostLoadOpCreator(
    OpDefinitions({{Onnx::CustomOperators::HostLoad, hostLoadOpDef}}),
    [](const OpCreatorInfo &info) {
      TensorId streamid =
          info.attributes.getAttribute<Attributes::String>("streamid");
      return std::unique_ptr<HostLoadOp>(
          new HostLoadOp(info.opid, info.settings, streamid));
    },
    true);

static OpDefinition hostStoreOpDef({OpDefinition::Inputs({{"X", T}}),
                                    OpDefinition::Outputs({}),
                                    OpDefinition::Attributes({})});

static OpCreator<HostStoreOp> hostStoreOpCreator(
    OpDefinitions({{Onnx::CustomOperators::HostStore, hostStoreOpDef}}),
    [](const OpCreatorInfo &info) {
      TensorId streamid =
          info.attributes.getAttribute<Attributes::String>("streamid");
      return std::unique_ptr<HostStoreOp>(
          new HostStoreOp(info.opid, info.settings, streamid));
    },
    true);
} // namespace
} // namespace popart
