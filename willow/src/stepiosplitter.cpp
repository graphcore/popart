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

SplitIOTensorInfo::SplitIOTensorInfo()
    : inIndex(0u), outIndex(0u), adapterMap{} {}

StepIOSplitter::StepIOSplitter(unsigned replicationFactor_)
    : replicationFactor(replicationFactor_), upstreamIo(nullptr),
      downstreamIoMap() {}

void StepIOSplitter::reset() {
  for (auto &entry1 : downstreamIoMap) {
    auto &splitIoTensorInfo    = entry1.second;
    splitIoTensorInfo.inIndex  = 0u;
    splitIoTensorInfo.outIndex = 0u;
    for (auto &entry2 : splitIoTensorInfo.adapterMap) {
      auto &adapter = entry2.second;
      adapter->reset();
    }
  }
}

void StepIOSplitter::setUpstreamIo(IStepIO *upstreamIo_) {
  upstreamIo = upstreamIo_;
  logging::devicex::debug("[StepIOSplitter] Reset upstream StepIO.");
  reset();
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

  unsigned lastInIndex    = 0;
  auto &splitIoTensorInfo = it->second;

  do {
    // Remember the index we're getting data for as it is used in the loop
    // condition and the value of inIndex may or may not change when we get
    // data.
    lastInIndex = splitIoTensorInfo.inIndex;

    // Determine if we are fetching for the requested index. If so, we'll want
    // to stick to the prefetch flag. If not, we are fetching data for an
    // earlier replica because we can't get data for the requested replica until
    // we have data for all preceding replicas.
    const bool isPrefetch =
        (splitIoTensorInfo.inIndex == replicationIndex) ? prefetch : false;

    auto adapterMapIt =
        splitIoTensorInfo.adapterMap.find(splitIoTensorInfo.inIndex);
    if (adapterMapIt != splitIoTensorInfo.adapterMap.end()) {

      // If we got data, populate downstream adapter and update state.
      // Get the downstream adapter.
      auto &adapter = adapterMapIt->second;

      logging::devicex::debug("[StepIOSplitter] Fetching input data for "
                              "tensor {}@{} {}",
                              id,
                              splitIoTensorInfo.inIndex,
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
        splitIoTensorInfo.inIndex =
            (splitIoTensorInfo.inIndex + 1) % replicationFactor;

        logging::devicex::debug(
            "[StepIOSplitter] Added data to back of input buffer for adapter "
            "{}@{} (buffer now has {} element(s))",
            id,
            splitIoTensorInfo.inIndex,
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
                  "tensor {}@{}",
                  id,
                  splitIoTensorInfo.inIndex);
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

  unsigned lastOutIndex   = 0;
  auto &splitIoTensorInfo = it->second;

  do {
    // Remember the index we're getting data for as it is updated in the loop.
    lastOutIndex = splitIoTensorInfo.outIndex;

    auto adapterMapIt =
        splitIoTensorInfo.adapterMap.find(splitIoTensorInfo.outIndex);
    if (adapterMapIt != splitIoTensorInfo.adapterMap.end()) {

      // If we got data, populate downstream adapter and update state.
      // Get the downstream adapter.
      auto &adapter = adapterMapIt->second;

      logging::devicex::debug("[StepIOSplitter] Getting output data for "
                              "tensor {}@{} adapter(s)",
                              id,
                              splitIoTensorInfo.outIndex);

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
        splitIoTensorInfo.outIndex =
            (splitIoTensorInfo.outIndex + 1) % replicationFactor;

        logging::devicex::debug(
            "[StepIOSplitter] Added data to back of output buffer for adapter "
            "{}@{} (buffer now has {} element(s))",
            id,
            splitIoTensorInfo.outIndex,
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
                  "tensor {}@{}",
                  id,
                  splitIoTensorInfo.outIndex);
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
    auto &splitIoTensorInfo = it1->second;
    auto it2 = splitIoTensorInfo.adapterMap.find(replicationIndex);
    if (it2 != splitIoTensorInfo.adapterMap.end()) {
      // We already have an adapter.
      return it2->second.get();
    } else {
      // We have a StepIOTensorInfo but no adapter for this replication index.
      auto &adapter = splitIoTensorInfo.adapterMap[replicationIndex];
      adapter       = std::make_unique<StepIOSplitterAdapter>(
          this, replicationIndex, id, info);
      return adapter.get();
    }
  } else {
    // We don't even have a StepIOTensorInfo for this tensor yet.
    auto &splitIoTensorInfo = downstreamIoMap[id];
    auto &adapter           = splitIoTensorInfo.adapterMap[replicationIndex];
    adapter                 = std::make_unique<StepIOSplitterAdapter>(
        this, replicationIndex, id, info);
    return adapter.get();
  }
}

} // namespace popart
