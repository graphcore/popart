// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/stepiosplitter.hpp>

namespace popart {

StepIOSplitterAdapter::StepIOSplitterAdapter(StepIOSplitter *splitter_,
                                             unsigned replicationIndex_,
                                             TensorId id,
                                             const TensorInfo &info)
    : splitter(splitter_), replicationIndex(replicationIndex_), adapterId(id),
      inData(), outData(), emptyVoidData{nullptr, info} {}

ConstVoidData
StepIOSplitterAdapter::in(TensorId id, int64_t numElements, bool prefetch) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  logging::devicex::debug("[StepIOSplitter] Input fetch for adapter {}@{} "
                          "(buffer has {} element(s))",
                          id,
                          replicationIndex,
                          inData.size());

  // If we have no data, ask for data.
  if (inData.empty()) {
    splitter->getInData(id, numElements, replicationIndex, prefetch);
  }

  // We should have data now, unless a prefetch didn't get data.
  if (!inData.empty()) {
    return inData.front();
  } else {
    if (prefetch) {
      return emptyVoidData;
    } else {
      throw error("Unable to fetch input data from upstream IStepIO");
    }
  }
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
    splitter->getOutData(id, numElements, replicationIndex);
  }

  // We should have data now. If not, it's a problem.
  if (!outData.empty()) {
    return outData.front();
  } else {
    throw error("Unable to fetch output data from upstream IStepIO");
  }
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
    : replicationFactor(replicationFactor_), inIndex(0u), outIndex(0u),
      upstreamIo(nullptr), downstreamIoMap() {}

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

void StepIOSplitter::getInData(TensorId id,
                               int64_t numElements,
                               unsigned replicationIndex,
                               bool prefetch) {

  // Check we have an upstream step io.
  if (!upstreamIo) {
    throw error("Upstream StepIO not set.");
  }

  // Check if we've got adapters for this tensor.
  auto it = downstreamIoMap.find(id);
  if (it == downstreamIoMap.end()) {
    throw error("No downstream StepIOs set");
  }

  unsigned lastInIndex = 0;
  auto &adapterMap     = it->second;

  do {
    // Remember the index we're getting data for as it is used in the loop
    // condition and the value of inIndex may or may not change when we get
    // data.
    lastInIndex = inIndex;

    // Determine if we are fetching for the requested index. If so, we'll want
    // to stick to the prefetch flag. If not, we are fetching data for an
    // earlier replica because we can't get data for the requested replica until
    // we have data for all preceding replicas.
    const bool isPrefetch = (inIndex == replicationIndex) ? prefetch : false;

    auto adapterMapIt = adapterMap.find(inIndex);
    if (adapterMapIt != adapterMap.end()) {

      // If we got data, populate downstream adapter and update state.
      // Get the downstream adapter.
      auto &adapter = adapterMapIt->second;

      logging::devicex::debug("[StepIOSplitter] Fetching input data for "
                              "tensor {} for {} adapter(s){}",
                              id,
                              replicationFactor,
                              isPrefetch ? " (prefetch)" : "");

      // Ask for data.
      const auto data = upstreamIo->in(id, numElements, isPrefetch);
      // Did get we data?
      const bool receivedData = (data.data != nullptr);

      if (receivedData) {
        // Store the data.
        adapter->getInData().push_back(data);
        // Mark the upsteam data as 'complete' so we can move on to the next
        // replicationIndex.
        upstreamIo->inComplete(id, numElements);
        // Update inData.
        inIndex = (inIndex + 1) % replicationFactor;

        logging::devicex::debug(
            "[StepIOSplitter] Added data to back of input buffer for adapter "
            "{}@{} (buffer now has {} element(s))",
            id,
            replicationIndex,
            adapter->getInData().size());
      } else {
        // If we didn't get data it's an error unless we are prefetching.
        if (!isPrefetch) {
          throw error("[StepIOSplitter] Upstream IStepIO unexpectedly did "
                      "not provide input data for tensor {}",
                      id);
        }
      }

    } else {
      // We can't do much without a downstream adapter.
      throw error("[StepIOSplitter] No downstream adapter found for input "
                  "tensor {}, replica {}",
                  id,
                  inIndex);
    }
  } while (lastInIndex != replicationIndex);
}

void StepIOSplitter::getOutData(TensorId id,
                                int64_t numElements,
                                unsigned replicationIndex) {

  // Check we have an upstream step io.
  if (!upstreamIo) {
    throw error("Upstream StepIO not set.");
  }

  // Check if we've got adapters for this tensor.
  auto it = downstreamIoMap.find(id);
  if (it == downstreamIoMap.end()) {
    throw error("No downstream StepIOs set");
  }

  unsigned lastOutIndex = 0;
  auto &adapterMap      = it->second;

  do {
    // Remember the index we're getting data for as it is updated in the loop.
    lastOutIndex = outIndex;

    auto adapterMapIt = adapterMap.find(outIndex);
    if (adapterMapIt != adapterMap.end()) {

      // If we got data, populate downstream adapter and update state.
      // Get the downstream adapter.
      auto &adapter = adapterMapIt->second;

      logging::devicex::debug("[StepIOSplitter] Getting output data for "
                              "tensor {} for {} adapter(s)",
                              id,
                              replicationFactor);

      // Ask for data.
      const auto data = upstreamIo->out(id, numElements);
      // Did get we data?
      const bool receivedData = (data.data != nullptr);

      if (receivedData) {
        // Store the data.
        adapter->getOutData().push_back(data);
        // Mark the upsteam data as 'complete' so we can move on to the next
        // replicationIndex.
        upstreamIo->outComplete(id);
        // Update inData.
        outIndex = (outIndex + 1) % replicationFactor;

        logging::devicex::debug(
            "[StepIOSplitter] Added data to back of output buffer for adapter "
            "{}@{} (buffer now has {} element(s))",
            id,
            replicationIndex,
            adapter->getOutData().size());
      } else {
        // This is always an error.
        throw error("[StepIOSplitter] Upstream IStepIO unexpectedly did not "
                    "provide output data for tensor {}",
                    id);
      }

    } else {
      // We can't do much without a downstream adapter.
      throw error("[StepIOSplitter] No downstream adapter found for output "
                  "tensor {}, replica {}",
                  id,
                  outIndex);
    }
  } while (lastOutIndex != replicationIndex);
}

void StepIOSplitter::assertNumElements(const Ir &ir) const {
  if (upstreamIo) {
    upstreamIo->assertNumElements(ir);
  } else {
    throw error("Upstream StepIO not set.");
  }
}

IStepIO *StepIOSplitter::getDownstreamStepIO(TensorId id,
                                             const TensorInfo &info,
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
      adapter       = std::make_unique<StepIOSplitterAdapter>(
          this, replicationIndex, id, info);
      return adapter.get();
    }
  } else {
    // Create new adapter in new replicaMap.
    auto &adapter = downstreamIoMap[id][replicationIndex];
    adapter       = std::make_unique<StepIOSplitterAdapter>(
        this, replicationIndex, id, info);
    return adapter.get();
  }
}

} // namespace popart
