// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/op/exchange/multiexchangex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

MultiExchangeOpx::MultiExchangeOpx(Op *op, Devicex *devicex)
    : ExchangeBaseOpx(op, devicex) {
  verifyOp<MultiExchangeOp>(op, Onnx::CustomOperators::MultiExchange);
}

InputCreatorType MultiExchangeOpx::getInputCreatorType(InIndex index) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  auto indices = multiExchangeOp.inIndexToDescriptorIndex(index);

  return (indices.second == 0 &&
          multiExchangeOp.getExchangeDescriptor(indices.first).getDirection() ==
              ExchangeDirection::Load)
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

bool MultiExchangeOpx::canUnwind(InIndex in, OutIndex out) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  auto inIndices  = multiExchangeOp.inIndexToDescriptorIndex(in);
  auto outIndices = multiExchangeOp.outIndexToDescriptorIndex(out);

  return inIndices.first == outIndices.first && inIndices.second == 0;
}

snap::Tensor MultiExchangeOpx::unwindTensorLayout(snap::Tensor tensor,
                                                  InIndex,
                                                  OutIndex) const {
  return tensor;
}

view::RegMap MultiExchangeOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

std::set<OpxGrowPartId>
MultiExchangeOpx::getInGrowPartIds(Tensor *inTensor) const {
  std::set<OpxGrowPartId> partIds;

  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  auto indices = multiExchangeOp.input->indices(inTensor);

  for (auto index : indices) {
    partIds.insert(multiExchangeOp.inIndexToDescriptorIndex(index).first);
  }
  return partIds;
}

OpxGrowPartId MultiExchangeOpx::getOutGrowPartId(Tensor *outTensor) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  if (outTensor->hasProducer() &&
      outTensor->getProducer()->id == multiExchangeOp.id) {
    return multiExchangeOp
        .outIndexToDescriptorIndex(
            multiExchangeOp.output->indices(outTensor).front())
        .first;
  }
  throw error("[MultiExchangeOp] Tensor {} is not consumed by {}",
              outTensor->id,
              multiExchangeOp.debugName());
}

std::vector<std::pair<int, int>> MultiExchangeOpx::getSegments() const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  std::vector<std::pair<int, int>> segments;

  // A resource identifier denoting stream tensor ID or remote buffer ID
  std::set<std::string> resourcesSeen;

  int last = 0;
  int i    = 0;
  // Segment exchanges (by resource use) if required.
  // Each exchange needs a "landing pad" tensor. If that landing pad is already
  // used by a previous exchange descriptor, the merged exchange has to be
  // segmented into multiple smaller exchanges so that the landing pad becomes
  // available again. Segmenting decreases potential IO/compute overlap.
  for (; i < multiExchangeOp.getNumExchanges(); ++i) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(i);
    auto resourceId = descriptor.getResourceId();
    if (resourcesSeen.find(resourceId) != resourcesSeen.end()) {
      logging::opx::info("[MultiExchangeOpx] Resource {} used more than once, "
                         "exchange will be segmented.",
                         descriptor.getResourceId());
      resourcesSeen.clear();
      segments.push_back({last, i});
      last = i;
    }
    resourcesSeen.insert(resourceId);
  }
  if (i != last) {
    segments.push_back({last, i});
  }

  return segments;
}

void MultiExchangeOpx::growPart(OpxGrowPartId id) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  MultiExchangeOpxState *state =
      dv_p->lowering().getOpxState<MultiExchangeOpxState>(multiExchangeOp.id);

  // Prepare descriptorx
  auto descriptor  = multiExchangeOp.getExchangeDescriptor(id);
  auto descriptorx = getExchangeDescriptorx(dv_p, descriptor);

  std::vector<std::pair<TensorId, snap::Tensor>> inTensors;

  // Get tensors
  for (auto input : multiExchangeOp.input->tensorIdMap()) {
    auto indices = multiExchangeOp.inIndexToDescriptorIndex(input.first);
    if (indices.first == id) {
      logging::opx::trace(
          "[MultiExchangeOpx] Adding tensor {} for descriptor {}",
          input.second,
          descriptor);
      inTensors.push_back({input.second, getInTensor(input.first)});
    }
  }

  // Set tensors
  descriptorx->setInTensors(inTensors);

  // Pre process the exchange
  logging::opx::debug("[MultiExchangeOpx] Growing pre-exchange for {}",
                      descriptor);
  descriptorx->pre(
      srcVirtualGraph(multiExchangeOp.descriptorIndexToInIndices(id).front()),
      state->preSeqs[id],
      debugContext());

  // Exchange
  logging::opx::debug("[MultiExchangeOpx] Growing exchange for {}", descriptor);
  descriptorx->exchange(
      srcVirtualGraph(multiExchangeOp.descriptorIndexToInIndices(id).front()),
      state->exchangeSeqs[id],
      debugContext());

  // Post process the exchange
  logging::opx::debug("[MultiExchangeOpx] Growing post-exchange for {}",
                      descriptor);
  descriptorx->post(
      srcVirtualGraph(multiExchangeOp.descriptorIndexToInIndices(id).front()),
      state->postSeqs[id],
      debugContext());

  // Set output view changers
  InIndex inIdx   = 0;
  OutIndex outIdx = 0;
  for (int i = 0; i < multiExchangeOp.getNumExchanges(); ++i) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(i);
    if (i == id && descriptor.getNumOutputs() >= 1) {
      if (hasInViewChangers(inIdx)) {
        setOutViewChangers(outIdx, getInViewChangers(inIdx));
      }
    }
    inIdx += descriptor.getNumInputs();
    outIdx += descriptor.getNumOutputs();
  }

  // Set output tensors
  for (auto output : multiExchangeOp.output->tensorIdMap()) {
    auto indices = multiExchangeOp.outIndexToDescriptorIndex(output.first);
    if (indices.first == id) {
      setOutTensor(output.first,
                   descriptorx->getOutTensors().at(indices.second));
    }
  }
}

void MultiExchangeOpx::grow(poplar::program::Sequence &prog) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  MultiExchangeOpxState *state =
      dv_p->lowering().getOpxState<MultiExchangeOpxState>(multiExchangeOp.id);

  auto segments = getSegments();

  logging::opx::debug("[MultiExchangeOpx] Exchange segmented: {}", segments);

  for (int j = 0; j < segments.size(); ++j) {
    // Prepare for the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.add(state->preSeqs[i]);
    }

    // Exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.add(state->exchangeSeqs[i]);
    }

    // Post process the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.add(state->postSeqs[i]);
    }
  }
}

snap::Graph &MultiExchangeOpx::inGraph(InIndex in) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    auto &multiExchangeOp = getOp<MultiExchangeOp>();
    auto vgid = multiExchangeOp.getIntrospectionInVirtualGraphId(in);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  } else {
    return dv_p->lowering().graph();
  }
}

snap::Graph &MultiExchangeOpx::outGraph(OutIndex out) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    auto &multiExchangeOp = getOp<MultiExchangeOp>();
    auto vgid = multiExchangeOp.getIntrospectionInVirtualGraphId(out);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  } else {
    return dv_p->lowering().graph();
  }
}

namespace {
OpxCreator<MultiExchangeOpx>
    MultiExchangeOpxCreator(Onnx::CustomOperators::MultiExchange);
} // namespace
} // namespace popx
} // namespace popart
