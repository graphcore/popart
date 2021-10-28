// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

RemoteStoreOp::RemoteStoreOp(const OperatorIdentifier &_opid,
                             const Op::Settings &settings_,
                             RemoteBufferId rbid_)
    : ExchangeBaseOp(_opid, settings_), remoteBufferId(rbid_) {}

std::unique_ptr<Op> RemoteStoreOp::clone() const {
  return std::make_unique<RemoteStoreOp>(*this);
}

void RemoteStoreOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remoteBufferId);
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

RemoteLoadInplaceOp::RemoteLoadInplaceOp(const OperatorIdentifier &_opid,
                                         const Op::Settings &settings_,
                                         RemoteBufferId rbid_)
    : ExchangeBaseOp(_opid, settings_), remoteBufferId(rbid_) {}

std::unique_ptr<Op> RemoteLoadInplaceOp::clone() const {
  return std::make_unique<RemoteLoadInplaceOp>(*this);
}

void RemoteLoadInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remoteBufferId);
}

void RemoteLoadInplaceOp::setup() {
  outInfo(getLocalTensorOutIndex()) = inInfo(getLocalTensorInIndex());
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
    throw error("Invalid index passed to RemoteLoadOp::aliases");
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

ReplicatedTensorShardingIndices
RemoteLoadInplaceOp::getReplicatedTensorShardingIndices() const {
  return {{{RemoteLoadInplaceOp::getLocalTensorInIndex()},
           {RemoteLoadInplaceOp::getLocalTensorOutIndex()}}};
}

ExchangeDescriptor RemoteLoadInplaceOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(ExchangeDirection::Load,
                            remoteBufferId,
                            settings.vgraphId,
                            settings.tileSet,
                            input->n(),
                            output->n());
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

static OpCreator<RemoteLoadInplaceOp> remoteLoadOpCreator(
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
