// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/error.hpp"
#include "popart/operators.hpp"
#include <algorithm>
#include <popart/alias/aliasmodel.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

// RemoteStoreOp
RemoteStoreOp::RemoteStoreOp(const OperatorIdentifier &_opid,
                             const Op::Settings &settings_,
                             RemoteBufferId rbid_)
    : RemoteBaseOp(_opid, settings_, rbid_) {}

std::unique_ptr<Op> RemoteStoreOp::clone() const {
  return std::make_unique<RemoteStoreOp>(*this);
}

ReplicatedTensorShardingIndices
RemoteStoreOp::getReplicatedTensorShardingIndices() const {
  return {{{RemoteStoreOp::getLocalTensorInIndex()}, {}}};
}

ExchangeDescriptor RemoteStoreOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Store,
                            remoteBufferId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n());
}

// RemoteLoadOp
RemoteLoadOp::RemoteLoadOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_,
                           RemoteBufferId rbid_)
    : RemoteBaseOp(_opid, settings_, rbid_) {}

std::unique_ptr<Op> RemoteLoadOp::clone() const {
  return std::make_unique<RemoteLoadOp>(*this);
}

void RemoteLoadOp::setup() {
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
}

ReplicatedTensorShardingIndices
RemoteLoadOp::getReplicatedTensorShardingIndices() const {
  return {{{RemoteLoadOp::getLocalTensorInIndex()},
           {RemoteLoadOp::getLocalTensorOutIndex()}}};
}

ExchangeDescriptor RemoteLoadOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            remoteBufferId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n(),
                            false);
}

std::vector<std::tuple<OperatorIdentifier, float>>
RemoteLoadOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::RemoteLoadInplace, 10}};
}

std::unique_ptr<Op>
RemoteLoadOp::getInplaceVariant(const OperatorIdentifier &operatorId) const {
  if (operatorId == Onnx::CustomOperators::RemoteLoadInplace) {
    return std::make_unique<RemoteLoadInplaceOp>(*this);
  }
  // Catch remaining cases and throw an error
  return Op::getInplaceVariant(operatorId);
}

void RemoteLoadOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier0(*this);
}

poprithms::memory::inplace::Proposal
RemoteLoadOp::mapInplaceProposal(const AliasModel &aliasModel,
                                 OperatorIdentifier opId) const {
  return mapInplaceProposalGate0(aliasModel, opId);
}

// RemoteLoadInplaceOp
RemoteLoadInplaceOp::RemoteLoadInplaceOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &settings_,
                                         RemoteBufferId rbid_)
    : RemoteLoadOp(_opid, settings_, rbid_) {}

RemoteLoadInplaceOp::RemoteLoadInplaceOp(const RemoteLoadOp &remoteLoadOp)
    : RemoteLoadOp(Onnx::CustomOperators::RemoteLoadInplace,
                   remoteLoadOp.getSettings(),
                   remoteLoadOp.getRemoteBufferId()) {}

std::unique_ptr<Op> RemoteLoadInplaceOp::clone() const {
  return std::make_unique<RemoteLoadInplaceOp>(*this);
}

view::Regions RemoteLoadInplaceOp::modifies(InIndex index) const {
  return aliases(index, 0);
}

view::Regions RemoteLoadInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getLocalTensorInIndex()) {
    return {view::Region::getFull(inShape(in), view::AccessType::Write)};
  } else if (in == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to RemoteLoadInplaceOp::aliases");
  }
}

view::RegMap RemoteLoadInplaceOp::fwdRegMap(InIndex inIndex,
                                            OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getLocalTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(outRank(getLocalTensorOutIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap RemoteLoadInplaceOp::bwdRegMap(InIndex inIndex,
                                            OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getLocalTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(inRank(getRemoteBufferOffsetInIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

std::vector<std::tuple<OperatorIdentifier, float>>
RemoteLoadInplaceOp::inplacePriorityDefault() const {
  return {};
}

std::unique_ptr<Op>
RemoteLoadInplaceOp::getInplaceVariant(const OperatorIdentifier &o) const {
  // This throws an error
  return Op::getInplaceVariant(o);
}

ExchangeDescriptor RemoteLoadInplaceOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            remoteBufferId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n(),
                            true);
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    remoteLoadOpDef({OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
                     OpDefinition::Outputs({{"Y", T}}),
                     OpDefinition::Attributes({})});

static OpDefinition remoteStoreOpDef(
    {OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
     OpDefinition::Outputs({}),
     OpDefinition::Attributes({})});

static OpCreator<RemoteLoadOp> remoteLoadOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteLoad, remoteLoadOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t bufferid =
          info.attributes.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<RemoteLoadOp>(
          new RemoteLoadOp(info.opid, info.settings, bufferid));
    },
    true);

static OpCreator<RemoteLoadInplaceOp> remoteLoadInplaceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteLoadInplace,
                    remoteLoadOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t bufferid =
          info.attributes.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<RemoteLoadInplaceOp>(
          new RemoteLoadInplaceOp(info.opid, info.settings, bufferid));
    },
    true);

static OpCreator<RemoteStoreOp> remoteStoreOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteStore, remoteStoreOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t bufferid =
          info.attributes.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<RemoteStoreOp>(
          new RemoteStoreOp(info.opid, info.settings, bufferid));
    },
    true);

} // namespace popart
