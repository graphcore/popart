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
    : HostBaseOp(_opid, settings_, sid_) {}

std::unique_ptr<Op> HostLoadOp::clone() const {
  return std::make_unique<HostLoadOp>(*this);
}

void HostLoadOp::setup() {
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
HostLoadOp::fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                                       const ReplEqInputMap &inputMap,
                                       ReplicaEqualAnalysisProxy &proxy) const {

  auto mode =
      getIr().getTensor(getHostStreamTensorId())->getReplicatedStreamMode();
  IsReplicaEqual value = (mode == ReplicatedStreamMode::Broadcast);

  // Prepare result map.
  ReplEqOutputMap result;
  for (auto &output : output->tensorMap()) {
    result[output.first] = value;
  }

  return {result, proxy.getModifiedInputMapFromAliases(this, result)};
}

std::vector<std::tuple<OperatorIdentifier, float>>
HostLoadOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::HostLoadInplace, 10}};
}

std::unique_ptr<Op>
HostLoadOp::getInplaceVariant(const OperatorIdentifier &operatorId) const {
  if (operatorId == Onnx::CustomOperators::HostLoadInplace) {
    return std::make_unique<HostLoadInplaceOp>(*this);
  }
  // Catch remaining cases and throw an error
  return Op::getInplaceVariant(operatorId);
}

void HostLoadOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier0(*this);
}

poprithms::memory::inplace::Proposal
HostLoadOp::mapInplaceProposal(const AliasModel &aliasModel,
                               OperatorIdentifier opId) const {
  return mapInplaceProposalGate0(aliasModel, opId);
}

ExchangeDescriptor HostLoadOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            getHostStreamTensorId(),
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n(),
                            false);
}

HostLoadInplaceOp::HostLoadInplaceOp(const OperatorIdentifier &_opid,
                                     const Op::Settings &settings_,
                                     TensorId sid_)
    : HostLoadOp(_opid, settings_, sid_) {}

HostLoadInplaceOp::HostLoadInplaceOp(const HostLoadOp &hostLoadOp)
    : HostLoadOp(Onnx::CustomOperators::HostLoadInplace,
                 hostLoadOp.getSettings(),
                 hostLoadOp.getHostStreamTensorId()) {}

std::unique_ptr<Op> HostLoadInplaceOp::clone() const {
  return std::make_unique<HostLoadInplaceOp>(*this);
}

void HostLoadInplaceOp::setup() {
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
}

view::Regions HostLoadInplaceOp::modifies(InIndex index) const {
  if (index == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else {
    throw error("Invalid index passed to HostLoadOp::modifies");
  }
}

view::Regions HostLoadInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else {
    throw error("Invalid index passed to HostLoadOp::aliases");
  }
}

view::RegMap HostLoadInplaceOp::fwdRegMap(InIndex inIndex,
                                          OutIndex outIndex) const {
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap HostLoadInplaceOp::bwdRegMap(InIndex inIndex,
                                          OutIndex outIndex) const {
  return Op::bwdRegMap(inIndex, outIndex);
}

std::unique_ptr<Op>
HostLoadInplaceOp::getInplaceVariant(const OperatorIdentifier &o) const {
  // This throws an error
  return Op::getInplaceVariant(o);
}

ExchangeDescriptor HostLoadInplaceOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            getHostStreamTensorId(),
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n(),
                            true);
}

HostStoreOp::HostStoreOp(const OperatorIdentifier &_opid,
                         const Op::Settings &settings_,
                         TensorId sid_)
    : HostBaseOp(_opid, settings_, sid_) {}

std::unique_ptr<Op> HostStoreOp::clone() const {
  return std::make_unique<HostStoreOp>(*this);
}

void HostStoreOp::setup() {
  if (!inInfo(HostStoreOp::getLocalTensorInIndex()).metaShape().empty()) {
    throw error(
        "RTS (replicated tensor sharded) tensor ({}) cannot be an input "
        "to HostStoreOp.",
        inTensor(HostStoreOp::getLocalTensorInIndex())->id);
  }
}

ExchangeDescriptor HostStoreOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Store,
                            getHostStreamTensorId(),
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n(),
                            false);
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
