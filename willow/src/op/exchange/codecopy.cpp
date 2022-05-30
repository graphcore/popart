// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <popart/op/exchange/codecopy.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/graphid.hpp"
#include "popart/ir.hpp"
#include "popart/op.hpp"
#include "popart/opserialiser.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
struct OperatorIdentifier;

RemoteCodeLoadOp::RemoteCodeLoadOp(const OperatorIdentifier &_opid,
                                   const GraphId &gid,
                                   const CodeMemoryType destinationType_,
                                   const Op::Settings &settings_)
    : ExchangeBaseOp(_opid, settings_), graphId(gid),
      destinationType(destinationType_) {
  if (!this->getIr().hasGraph(gid)) {
    throw error("Graph for GraphId {} for {} {} not found",
                gid,
                typeid(*this).name(),
                this->debugName());
  }
  settings.schedulePriority = std::numeric_limits<double>::lowest();
}

void RemoteCodeLoadOp::setup() {
  if (destinationType == CodeMemoryType::ExecutableMemory &&
      settings.tileSet == TileSet::Compute) {
    ; // Supported
  } else {
    throw error("destinationType {} and destination tileset {} is not "
                "supported for {} {}",
                destinationType,
                settings.tileSet,
                typeid(*this).name(),
                this->debugName());
  }
}

std::unique_ptr<Op> RemoteCodeLoadOp::clone() const {
  return std::make_unique<RemoteCodeLoadOp>(*this);
}

void RemoteCodeLoadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("gid", graphId);
  os.appendAttribute("destinationType", destinationType);
}

ExchangeDescriptor RemoteCodeLoadOp::getExchangeDescriptor(int index) const {
  return ExchangeDescriptor(
      ExchangeDirection::Load, graphId, settings.tileSet, destinationType);
}

static OpDefinition RemoteCodeLoadOpDef(
    {OpDefinition::Inputs(),
     OpDefinition::Outputs(),
     OpDefinition::Attributes({{"graphId", {"*"}},
                               {"destinationCodeMemoryType", {"*"}}})});

static OpCreator<RemoteCodeLoadOp> RemoteCodeLoadOpCreator(
    OpDefinitions({{Onnx::CustomOperators::RemoteCodeLoad,
                    RemoteCodeLoadOpDef}}),
    [](const OpCreatorInfo &info) {
      GraphId graphId =
          info.attributes.getAttribute<Attributes::String>("graphId");
      int destinationCodeMemoryType =
          info.attributes.getAttribute<Attributes::Int>(
              "destinationCodeMemoryType");
      return std::unique_ptr<RemoteCodeLoadOp>(new RemoteCodeLoadOp(
          info.opid,
          graphId,
          static_cast<CodeMemoryType>(destinationCodeMemoryType),
          info.settings));
    },
    true);

void InternalCodeCopyOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("gid", graphId);
  os.appendAttribute("source", source);
  os.appendAttribute("sourceType", sourceType);
  os.appendAttribute("destinationType", destinationType);
}

} // namespace popart
