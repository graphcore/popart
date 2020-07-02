// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/stepiosplitter.hpp>

namespace popart {

StepIOSplitterAdapter::StepIOSplitterAdapter(StepIOSplitter *splitter_,
                                             unsigned replicationIndex_,
                                             TensorId id)
    : splitter(splitter_), replicationIndex(replicationIndex_), adapterId(id),
      inData(), outData() {}

ConstVoidData
StepIOSplitterAdapter::in(TensorId id, int64_t numElements, bool) {
  ConstVoidData result;

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  // NOTE: We ignore the prefetch flag (as does StepIOGeneric::in).
  logging::devicex::debug("[StepIOSplitter] Input fetch for adapter {}@{} "
                          "(buffer has {} element(s))",
                          id,
                          replicationIndex,
                          inData.size());

  // If we have no data, ask for data.
  if (inData.empty()) {
    splitter->getInData(id, numElements);
  }

  // We should have data now. If not, it's a problem.
  if (!inData.empty()) {
    result = inData.front();
  } else {
    throw error("Unable to fetch input data from upstream IStepIO");
  }

  return result;
}

void StepIOSplitterAdapter::inComplete(TensorId id, int64_t numElements) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  if (!inData.empty()) {
    inData.pop_front();
    logging::devicex::debug(
        "[StepIOSplitter] Input read complete; discarded front of input buffer "
        "for adapter {}@{} (buffer now has {} element(s))",
        id,
        replicationIndex,
        inData.size());
  } else {
    throw error("No input data available to mark as complete");
  }
}

MutableVoidData StepIOSplitterAdapter::out(TensorId id, int64_t numElements) {
  MutableVoidData result;

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  logging::devicex::debug("[StepIOSplitter] Output fetch for adapter {}@{} "
                          "(buffer has {} element(s))",
                          id,
                          replicationIndex,
                          outData.size());

  // If we have no data, ask for data.
  if (outData.empty()) {
    splitter->getOutData(id, numElements);
  }

  // We should have data now. If not, it's a problem.
  if (!outData.empty()) {
    result = outData.front();
  } else {
    throw error("Unable to fetch output data from upstream IStepIO");
  }

  return result;
}

void StepIOSplitterAdapter::outComplete(TensorId id) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  if (!outData.empty()) {
    outData.pop_front();
    logging::devicex::debug(
        "[StepIOSplitter] Output write complete; discarded front of output "
        "buffer for adapter {}@{} (buffer now has {} elements)",
        id,
        replicationIndex,
        outData.size());
  } else {
    throw error("No output data available to mark as complete");
  }
}

void StepIOSplitterAdapter::assertNumElements(const Ir &ir) const {
  splitter->assertNumElements(ir);
}

void StepIOSplitterAdapter::reset() {
  inData.clear();
  outData.clear();
}

StepIOSplitter::StepIOSplitter(unsigned replicationFactor_)
    : replicationFactor(replicationFactor_), upstreamIo(nullptr),
      downstreamIoMap() {}

void StepIOSplitter::reset() {
  for (auto &entry1 : downstreamIoMap) {
    for (auto &entry2 : entry1.second) {
      auto &adapter = entry2.second;
      adapter->reset();
    }
  }
}

void StepIOSplitter::reset(IStepIO *upstreamIo_) {
  logging::devicex::debug("[StepIOSplitter] Reset");
  reset();
  upstreamIo = upstreamIo_;
}

void StepIOSplitter::getInData(TensorId id, int64_t numElements) {
  if (upstreamIo) {
    // Check if we've got adapters for this tensor.
    auto it = downstreamIoMap.find(id);
    if (it != downstreamIoMap.end()) {
      // Check the number of replicas is correct.
      auto numReplicas = it->second.size();
      if (numReplicas == replicationFactor) {
        logging::devicex::debug("[StepIOSplitter] Getting input data for "
                                "tensor {} for {} adapter(s)",
                                id,
                                replicationFactor);
        // Get in data for each replica.
        for (auto &entry : it->second) {
          auto &adapter = entry.second;
          auto data     = upstreamIo->in(id, numElements, false);
          upstreamIo->inComplete(id, numElements);
          adapter->getInData().push_back(data);

          logging::devicex::debug(
              "[StepIOSplitter] Added data to back of input buffer for adapter "
              "{}@{} (buffer now has {} element(s))",
              it->first,
              entry.first,
              adapter->getInData().size());
        }
      } else {
        throw error("{} downstream StepIOs set (expected {})",
                    numReplicas,
                    replicationFactor);
      }
    } else {
      throw error("No downstream StepIOs set");
    }
  } else {
    throw error("Upstream StepIO not set.");
  }
}

void StepIOSplitter::getOutData(TensorId id, int64_t numElements) {
  if (upstreamIo) {
    // Check if we've got adapters for this tensor.
    auto it = downstreamIoMap.find(id);
    if (it != downstreamIoMap.end()) {
      // Check the number of replicas is correct.
      auto numReplicas = it->second.size();
      if (numReplicas == replicationFactor) {
        logging::devicex::debug("[StepIOSplitter] Getting output data "
                                "locations for tensor {} for {} adapter(s)",
                                id,
                                replicationFactor);
        // Get out data for each replica.
        for (auto &entry : it->second) {
          auto &adapter = entry.second;
          auto data     = upstreamIo->out(id, numElements);
          upstreamIo->outComplete(id);
          adapter->getOutData().push_back(data);
          logging::devicex::debug(
              "[StepIOSplitter] Added data location to back of output buffer "
              "for adapter {}@{} (buffer now has {} element(s))",
              it->first,
              entry.first,
              adapter->getOutData().size());
        }
      } else {
        throw error("{} downstream StepIOs set (expected {})",
                    numReplicas,
                    replicationFactor);
      }
    } else {
      throw error("No downstream StepIOs set");
    }
  } else {
    throw error("Upstream StepIO not set.");
  }
}

void StepIOSplitter::assertNumElements(const Ir &ir) const {
  if (upstreamIo) {
    upstreamIo->assertNumElements(ir);
  } else {
    throw error("Upstream StepIO not set.");
  }
}

IStepIO *StepIOSplitter::getDownstreamStepIO(TensorId id,
                                             unsigned replicationIndex) {
  auto it1 = downstreamIoMap.find(id);
  if (it1 != downstreamIoMap.end()) {
    auto &replicaMap = it1->second;
    auto it2         = replicaMap.find(replicationIndex);
    if (it2 != replicaMap.end()) {
      // Return an adapter we have already.
      return it2->second.get();
    } else {
      // Create a new adapter for replication index for existing replicaMap.
      auto &adapter = replicaMap[replicationIndex];
      adapter =
          std::make_unique<StepIOSplitterAdapter>(this, replicationIndex, id);
      return adapter.get();
    }
  } else {
    // Create new adapter in new replicaMap.
    auto &adapter = downstreamIoMap[id][replicationIndex];
    adapter =
        std::make_unique<StepIOSplitterAdapter>(this, replicationIndex, id);
    return adapter.get();
  }
}

} // namespace popart
