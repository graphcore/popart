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

void MultiExchangeOpx::grow(poplar::program::Sequence &prog) const {
  auto &multiExchangeOp = getOp<MultiExchangeOp>();

  std::vector<std::shared_ptr<ExchangeDescriptorx>> descriptorxs;

  // Prepare descriptorx
  for (int i = 0; i < multiExchangeOp.getNumExchanges(); ++i) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(i);
    descriptorxs.push_back(getExchangeDescriptorx(dv_p, descriptor));
  }

  std::vector<std::vector<std::pair<TensorId, snap::Tensor>>> tensors(
      descriptorxs.size());

  // Get tensors
  for (auto input : multiExchangeOp.input->tensorIdMap()) {
    auto indices = multiExchangeOp.inIndexToDescriptorIndex(input.first);
    tensors.at(indices.first)
        .push_back({input.second, getInTensor(input.first)});
  }

  // Set tensors
  for (int i = 0; i < multiExchangeOp.getNumExchanges(); ++i) {
    descriptorxs.at(i)->setInTensors(tensors.at(i));
  }

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

  logging::opx::debug("[MultiExchangeOpx] Exchange segmented: {}", segments);

  for (int j = 0; j < segments.size(); ++j) {
    // Prepare for the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      logging::opx::debug("[MultiExchangeOpx] Growing pre-exchange for {}",
                          descriptorxs.at(i));
      descriptorxs.at(i)->pre(
          srcVirtualGraph(
              multiExchangeOp.descriptorIndexToInIndices(i).front()),
          prog,
          debugContext());
    }

    // Exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      logging::opx::debug("[MultiExchangeOpx] Growing exchange for {}",
                          descriptorxs.at(i));
      descriptorxs.at(i)->exchange(
          srcVirtualGraph(
              multiExchangeOp.descriptorIndexToInIndices(i).front()),
          prog,
          debugContext());
    }

    // Post process the exchange
    for (int i = segments.at(j).first; i < segments.at(j).second; ++i) {
      logging::opx::debug("[MultiExchangeOpx] Growing post-exchange for {}",
                          descriptorxs.at(i));
      descriptorxs.at(i)->post(
          srcVirtualGraph(
              multiExchangeOp.descriptorIndexToInIndices(i).front()),
          prog,
          debugContext());
    }
  }

  // Set output view changers
  InIndex inIdx   = 0;
  OutIndex outIdx = 0;
  for (int i = 0; i < multiExchangeOp.getNumExchanges(); ++i) {
    auto descriptor = multiExchangeOp.getExchangeDescriptor(i);
    if (descriptor.getNumOutputs() >= 1) {
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
    setOutTensor(
        output.first,
        descriptorxs.at(indices.first)->getOutTensors().at(indices.second));
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
