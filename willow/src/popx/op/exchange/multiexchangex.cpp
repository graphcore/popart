// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/op/exchange/multiexchangex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/util.hpp"

namespace snap {

class Graph;
} // namespace snap

namespace popart {

namespace popx {

namespace {

snap::program::Sequence &
get(std::map<int, snap::program::Sequence> &xs, int index, snap::Graph &graph) {
  auto found = xs.find(index);
  if (found == xs.end()) {
    xs.insert({index, graph});
  }
  return xs.at(index);
}

} // namespace

MultiExchangeOpx::MultiExchangeOpx(Op *op, Devicex *devicex)
    : ExchangeBaseOpx(op, devicex) {
  verifyOp<MultiExchangeOp>(op, Onnx::CustomOperators::MultiExchange);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

InputCreatorType MultiExchangeOpx::getInputCreatorType(InIndex index) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  auto indices = multiExchangeOp.inIndexToDescriptorIndex(index);
  auto direction =
      multiExchangeOp.getExchangeDescriptor(indices.first).getDirection();

  if (indices.second == 0) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(indices.first);
    std::shared_ptr<ExchangeDescriptorx> descriptorx =
        getExchangeDescriptorx(dv_p, descriptor);
    if (descriptorx->rearrangeOnHost() ||
        descriptor.getTileSet() == TileSet::Compute) {
      // If rearranging on host or not using IO tiles, then use unwinding to
      // minimize rearrangements

      // `Unwind`: In most cases, the input tensor layout can be unwound from
      // the output, which will cause fewer on-device rearrangements
      // Only load operations have an output tensor to unwind from
      return direction == ExchangeDirection::Load ? InputCreatorType::CanUnwind
                                                  : InputCreatorType::Deadend;
    } else {
      // If rearranging on device and using IO tiles, use host transferrable
      // tensors to facilitate overlapped IO/compute

      // `CanCreate`: Create the tensor with createHostTransferrableTensor to
      // avoid blocking overlapped IO with misplaced inter-tile exchanges on
      // IO tiles
      // `Unwind`: Fallback if creating the tensor is not possible
      // Only load operations have an output tensor to unwind from
      return direction == ExchangeDirection::Load
                 ? InputCreatorType::CanCreateOrUnwind
                 : InputCreatorType::CanCreate;
    }
  }

  return PopOpx::getInputCreatorType(index);
}

bool MultiExchangeOpx::canUnwind(InIndex in, OutIndex out) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  auto inIndices  = multiExchangeOp.inIndexToDescriptorIndex(in);
  auto outIndices = multiExchangeOp.outIndexToDescriptorIndex(out);

  return inIndices.first == outIndices.first && inIndices.second == 0;
}

snap::Tensor MultiExchangeOpx::unwindTensorLayout(snap::Tensor tensor,
                                                  InIndex in,
                                                  OutIndex) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();
  auto inIndices        = multiExchangeOp.inIndexToDescriptorIndex(in);
  auto descriptor  = multiExchangeOp.getExchangeDescriptor(inIndices.first);
  auto descriptorx = getExchangeDescriptorx(dv_p, descriptor);

  return descriptorx->unwind(srcVirtualGraph(in), tensor);
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
  std::set<std::pair<std::string, ExchangeDirection>> resourcesSeen;

  int last = 0;
  int i    = 0;
  // Segment exchanges (by resource use) if required.
  // Each exchange needs a "landing pad" tensor. If that landing pad is already
  // used by a previous exchange descriptor, the merged exchange has to be
  // segmented into multiple smaller exchanges so that the landing pad becomes
  // available again. Segmenting decreases potential IO/compute overlap.
  for (; i < multiExchangeOp.getNumExchanges(); ++i) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(i);
    auto resourceId =
        std::make_pair(descriptor.getResourceId(), descriptor.getDirection());
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
      popart::popx::get(state->preSeqs, id, graph()),
      debugContext());

  // Exchange
  logging::opx::debug("[MultiExchangeOpx] Growing exchange for {}", descriptor);
  descriptorx->exchange(
      srcVirtualGraph(multiExchangeOp.descriptorIndexToInIndices(id).front()),
      popart::popx::get(state->exchangeSeqs, id, graph()),
      debugContext());

  // Post process the exchange
  logging::opx::debug("[MultiExchangeOpx] Growing post-exchange for {}",
                      descriptor);
  descriptorx->post(
      srcVirtualGraph(multiExchangeOp.descriptorIndexToInIndices(id).front()),
      popart::popx::get(state->postSeqs, id, graph()),
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

void MultiExchangeOpx::grow(snap::program::Sequence &prog) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  MultiExchangeOpxState *state =
      dv_p->lowering().getOpxState<MultiExchangeOpxState>(multiExchangeOp.id);

  auto segments = getSegments();

  logging::opx::debug("[MultiExchangeOpx] Exchange segmented: {}", segments);

  for (int j = 0; j < segments.size(); ++j) {
    // Prepare for the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.getPoplarSequence().add(
          popart::popx::get(state->preSeqs, i, graph()));
    }

    // Exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.getPoplarSequence().add(
          popart::popx::get(state->exchangeSeqs, i, graph()));
    }

    // Post process the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      prog.getPoplarSequence().add(
          popart::popx::get(state->postSeqs, i, graph()));
    }
  }
}

snap::Tensor
MultiExchangeOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  auto &multiExchangeOp                            = getOp<MultiExchangeOp>();
  std::shared_ptr<ExchangeDescriptorx> descriptorx = getExchangeDescriptorx(
      dv_p,
      multiExchangeOp.getExchangeDescriptor(
          multiExchangeOp.inIndexToDescriptorIndex(index).first));
  return descriptorx->create(inGraph(index), inInfo(index));
}

namespace {
OpxCreator<MultiExchangeOpx>
    MultiExchangeOpxCreator(Onnx::CustomOperators::MultiExchange);
} // namespace
} // namespace popx
} // namespace popart
