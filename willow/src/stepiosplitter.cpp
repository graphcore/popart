// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <stepiosplitter.hpp>

namespace popart {
namespace popx {
class Executablex;
}

StepIOSplitterAdapter::StepIOSplitterAdapter(StepIOSplitter *splitter_,
                                             SplitIOTensorInfo *tensorInfo_,
                                             unsigned replicationIndex_,
                                             unsigned replicationFactor_,
                                             TensorId id,
                                             const TensorInfo &info,
                                             const int maxInFetches_,
                                             const int maxOutFetches_)
    : splitter(splitter_), tensorInfo(tensorInfo_),
      replicationIndex(replicationIndex_),
      replicationFactor(replicationFactor_), adapterId(id),
      maxInFetches(maxInFetches_), maxOutFetches(maxOutFetches_), inData(),
      numInFetches(0u), numInIncompleteDownstream(0u),
      numInIncompleteUpstream(0u), outData(), numOutFetches(0u),
      numOutIncompleteDownstream(0u),
      numOutIncompleteUpstream(0u), emptyVoidData{nullptr, info} {}

ConstVoidData
StepIOSplitterAdapter::in(TensorId id, int64_t numElements, bool prefetch) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  // If we have no data, ask for data.
  if (inData.empty()) {
    inLog("Received Poplar callback 'in' with no input buffer from IStepIO "
          "already cached");
    splitter->getInData(id, numElements, replicationIndex, prefetch);
  } else {
    inLog("Received Poplar callback 'in' with input buffer from IStepIO "
          "already cached");
  }

  // We should have data now, unless a prefetch didn't get data.
  if (!inData.empty()) {
    const ConstVoidData result = inData.front();
    inData.pop_front();
    // Remember poplar hasn't completed this.
    numInIncompleteDownstream++;
    inLog("Returning input buffer to Poplar from adapter's cache");
    return result;
  } else {
    if (prefetch) {
      return emptyVoidData;
    } else {
      throw error("Unable to fetch input data from IStepIO");
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

  if (numInIncompleteDownstream > 0) {
    numInIncompleteDownstream--;
    numInIncompleteUpstream++;
    inLog("Received Poplar callback to 'inComplete'");
    splitter->inCompletionCallback(id, numElements, replicationIndex);
  } else {
    throw error("StepIOSplitterAdapter no data to complete for tensor {}", id);
  }
}

MutableVoidData StepIOSplitterAdapter::out(TensorId id, int64_t numElements) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  // If we have no data, ask for data.
  if (outData.empty()) {
    outLog("Received Poplar callback 'out' with no output buffer from IStepIO "
           "already cached");
    splitter->getOutData(id, numElements, replicationIndex);
  } else {
    outLog("Received Poplar callback 'out' with output buffer from IStepIO "
           "already cached");
  }

  // We should have data now. If not, it's a problem.
  if (!outData.empty()) {
    const MutableVoidData result = outData.front();
    outData.pop_front();
    // Remember poplar hasn't completed this.
    numOutIncompleteDownstream++;
    outLog("Returning output buffer to Poplar from adapter's cache");
    return result;
  } else {
    throw error("Unable to fetch output data from IStepIO");
  }
}

void StepIOSplitterAdapter::outComplete(TensorId id) {

  if (id != adapterId) {
    throw error("StepIOSplitterAdapter was created for tensor {} but used for "
                "tensor {}",
                adapterId,
                id);
  }

  if (numOutIncompleteDownstream > 0) {
    numOutIncompleteDownstream--;
    numOutIncompleteUpstream++;
    outLog("Received Poplar callback to 'outComplete'");
    splitter->outCompletionCallback(id, replicationIndex);
  } else {
    throw error("StepIOSplitterAdapter no data to complete for tensor {}", id);
  }
}

void StepIOSplitterAdapter::assertNumElements(
    const popx::Executablex &exe) const {
  splitter->assertNumElements(exe);
}

void StepIOSplitterAdapter::inLog(const char *action) const {

  if (tensorInfo && logging::devicex::isEnabled(logging::Level::Trace)) {
    std::stringstream logSs;
    logSs << "[StepIOSplitter] ";
    // Include adapter handle
    // <tensor>@in:<replicationIndex>/<replicationFactor>.
    logSs << "[" << adapterId << "@in:" << replicationIndex << "/"
          << replicationFactor;
    // Add adapter states.
    logSs << " - ";
    bool first = true;

    for (auto &entry : tensorInfo->adapterMap) {
      auto &repl = entry.first;
      auto &adap = entry.second;

      if (!first) {
        logSs << " ";
      }

      logSs << repl << "/" << replicationFactor << ":" << adap->numInFetches
            << "," << adap->inData.size() << ","
            << adap->numInIncompleteDownstream << ","
            << adap->numInIncompleteUpstream;

      first = false;
    }
    logSs << "] ";
    // Add the actual log event.
    logSs << action;

    logging::devicex::trace(logSs.str());
  }
}

void StepIOSplitterAdapter::outLog(const char *action) const {

  if (tensorInfo && logging::devicex::isEnabled(logging::Level::Debug)) {
    std::stringstream logSs;
    logSs << "[StepIOSplitter] ";
    // Include adapter handle
    // <tensor>@out:<replicationIndex>/<replicationFactor>.
    logSs << "[" << adapterId << "@out:" << replicationIndex << "/"
          << replicationFactor;

    if (logging::devicex::isEnabled(logging::Level::Trace)) {
      // Add adapter states when tracing.
      logSs << " - ";
      bool first = true;

      for (auto &entry : tensorInfo->adapterMap) {
        auto &repl = entry.first;
        auto &adap = entry.second;

        if (!first) {
          logSs << " ";
        }

        logSs << repl << "/" << replicationFactor << ":" << adap->numOutFetches
              << "," << adap->outData.size() << ","
              << adap->numOutIncompleteDownstream << ","
              << adap->numOutIncompleteUpstream;

        first = false;
      }
    }

    logSs << "] ";
    // Add the actual log event.
    logSs << action;

    logging::devicex::debug(logSs.str());
  }
}

bool StepIOSplitterAdapter::canAddInBuffer() const {
  return numInFetches < maxInFetches;
}

bool StepIOSplitterAdapter::canAddOutBuffer() const {
  return numOutFetches < maxOutFetches;
}

void StepIOSplitterAdapter::addInBuffer(const ConstVoidData &buf) {
  inData.push_back(buf);
  numInFetches++;
  inLog("Added an input buffer from IStepIO to adapter's cache");
}

void StepIOSplitterAdapter::addOutBuffer(const MutableVoidData &buf) {
  outData.push_back(buf);
  numOutFetches++;
  outLog("Added an output buffer from IStepIO to adapter's cache");
}

bool StepIOSplitterAdapter::tryInCompleteUpstream() {
  if (numInIncompleteUpstream > 0) {
    numInIncompleteUpstream--;
    return true;
  }
  return false;
}

bool StepIOSplitterAdapter::tryOutCompleteUpstream() {
  if (numOutIncompleteUpstream > 0) {
    numOutIncompleteUpstream--;
    return true;
  }
  return false;
}

void StepIOSplitterAdapter::reset() {
  inData.clear();
  outData.clear();
  numInFetches               = 0;
  numInIncompleteDownstream  = 0;
  numInIncompleteUpstream    = 0;
  numOutFetches              = 0;
  numOutIncompleteDownstream = 0;
  numOutIncompleteUpstream   = 0;
}

SplitIOTensorInfo::SplitIOTensorInfo()
    : inIndex(0u), inCompleteIndex(0u), outIndex(0u),
      outCompleteIndex(0u), adapterMap{} {}

StepIOSplitter::StepIOSplitter(
    unsigned replicationFactor_,
    std::function<unsigned(const TensorId &)> maxInFetchesPerReplFun_,
    std::function<unsigned(const TensorId &)> maxOutFetchesPerReplFun_)
    : replicationFactor(replicationFactor_),
      maxInFetchesPerReplFun(maxInFetchesPerReplFun_),
      maxOutFetchesPerReplFun(maxOutFetchesPerReplFun_), upstreamIo(nullptr),
      downstreamIoMap() {}

void StepIOSplitter::reset() {
  for (auto &entry1 : downstreamIoMap) {
    auto &splitIoTensorInfo            = entry1.second;
    splitIoTensorInfo.inIndex          = 0u;
    splitIoTensorInfo.inCompleteIndex  = 0u;
    splitIoTensorInfo.outIndex         = 0u;
    splitIoTensorInfo.outCompleteIndex = 0u;
    for (auto &entry2 : splitIoTensorInfo.adapterMap) {
      auto &adapter = entry2.second;
      adapter->reset();
    }
  }
  logging::devicex::info("[StepIOSplitter] Resetting StepIOSplitter.");
}

void StepIOSplitter::setUpstreamIo(IStepIO *upstreamIo_) {
  upstreamIo = upstreamIo_;
  logging::devicex::trace("[StepIOSplitter] Reset StepIO.");
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

      if (adapter->canAddInBuffer()) {
        // Log it.
        if (isPrefetch) {
          adapter->inLog(
              "Going to try fetching input buffer from IStepIO (prefetch)");
        } else {
          adapter->inLog("Going to try fetching input buffer from IStepIO");
        }

        // Ask for data.
        const auto data = upstreamIo->in(id, numElements, isPrefetch);
        // Did get we data?
        const bool receivedData = (data.data != nullptr);

        if (receivedData) {
          // Store the data.
          adapter->addInBuffer(data);
          // Update inData.
          splitIoTensorInfo.inIndex =
              (splitIoTensorInfo.inIndex + 1) % replicationFactor;
        } else {
          // If we didn't get data it's an error unless we are prefetching.
          if (!isPrefetch) {
            throw error("[StepIOSplitter] IStepIO unexpectedly did "
                        "not provide input data for tensor {}",
                        id);
          }
        }
      } else {
        adapter->inLog("Unable to fetch input buffer; reached maximum number "
                       "for this run");
        break;
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

      if (adapter->canAddOutBuffer()) {
        // Log it.
        adapter->outLog("Going to try and fetch output buffer from IStepIO");

        // Ask for data.
        const auto data = upstreamIo->out(id, numElements);
        // Did get we data?
        const bool receivedData = (data.data != nullptr);

        if (receivedData) {
          // Store the data.
          adapter->addOutBuffer(data);

          // Update inData.
          splitIoTensorInfo.outIndex =
              (splitIoTensorInfo.outIndex + 1) % replicationFactor;
        } else {
          // This is always an error.
          throw error("[StepIOSplitter] IStepIO unexpectedly did not "
                      "provide output data for tensor {}",
                      id);
        }
      } else {
        adapter->outLog("Unable to fetch output buffer; reached maximum number "
                        "for this run");
        break;
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

void StepIOSplitter::assertNumElements(const popx::Executablex &exe) const {
  if (upstreamIo) {
    upstreamIo->assertNumElements(exe);
  } else {
    throw error("Upstream StepIO not set.");
  }
}

IStepIO *StepIOSplitter::getDownstreamStepIO(TensorId id,
                                             const TensorInfo &info,
                                             unsigned replicationIndex) {
  const auto maxInFetches  = maxInFetchesPerReplFun(id);
  const auto maxOutFetches = maxOutFetchesPerReplFun(id);
  auto it1                 = downstreamIoMap.find(id);
  if (it1 != downstreamIoMap.end()) {
    auto &splitIoTensorInfo = it1->second;
    auto it2 = splitIoTensorInfo.adapterMap.find(replicationIndex);
    if (it2 != splitIoTensorInfo.adapterMap.end()) {
      // We already have an adapter.
      return it2->second.get();
    } else {
      // We have a StepIOTensorInfo but no adapter for this replication index.
      auto &adapter = splitIoTensorInfo.adapterMap[replicationIndex];
      adapter       = std::make_unique<StepIOSplitterAdapter>(this,
                                                        &splitIoTensorInfo,
                                                        replicationIndex,
                                                        replicationFactor,
                                                        id,
                                                        info,
                                                        maxInFetches,
                                                        maxOutFetches);
      return adapter.get();
    }
  } else {
    // We don't even have a StepIOTensorInfo for this tensor yet.
    auto &splitIoTensorInfo = downstreamIoMap[id];
    auto &adapter           = splitIoTensorInfo.adapterMap[replicationIndex];
    adapter                 = std::make_unique<StepIOSplitterAdapter>(this,
                                                      &splitIoTensorInfo,
                                                      replicationIndex,
                                                      replicationFactor,
                                                      id,
                                                      info,
                                                      maxInFetches,
                                                      maxOutFetches);
    return adapter.get();
  }
}

void StepIOSplitter::inCompletionCallback(TensorId id,
                                          int64_t numElements,
                                          unsigned replicationIndex) {
  // Check we have an upstream step io.
  if (!upstreamIo) {
    throw error("Upstream StepIO not set.");
  }

  auto it1 = downstreamIoMap.find(id);
  if (it1 == downstreamIoMap.end()) {
    throw error("No downstream StepIOs set");
  }

  auto &splitIoTensorInfo = it1->second;
  auto &inCompleteIndex   = splitIoTensorInfo.inCompleteIndex;

  while (true) {
    auto it2 = splitIoTensorInfo.adapterMap.find(inCompleteIndex);

    if (it2 != splitIoTensorInfo.adapterMap.end()) {
      auto &adapter = it2->second;
      if (adapter->tryInCompleteUpstream()) {
        upstreamIo->inComplete(id, numElements);
        adapter->inLog("Called 'inComplete' on IStepIO");
      } else {
        // Can't complete the next adapter.
        adapter->inLog("Not yet able to call 'inComplete' on IStepIO");
        break;
      }
    } else {
      // We can't do much without a downstream adapter.
      throw error("[StepIOSplitter] No downstream adapter found for input "
                  "tensor {}@{}",
                  id,
                  splitIoTensorInfo.outIndex);
    }

    // Try the next index.
    inCompleteIndex = (inCompleteIndex + 1) % replicationFactor;
  }
}

void StepIOSplitter::outCompletionCallback(TensorId id,
                                           unsigned replicationIndex) {
  // Check we have an upstream step io.
  if (!upstreamIo) {
    throw error("Upstream StepIO not set.");
  }

  auto it1 = downstreamIoMap.find(id);
  if (it1 == downstreamIoMap.end()) {
    throw error("No downstream StepIOs set");
  }

  auto &splitIoTensorInfo = it1->second;
  auto &outCompleteIndex  = splitIoTensorInfo.outCompleteIndex;

  while (true) {
    auto it2 = splitIoTensorInfo.adapterMap.find(outCompleteIndex);

    if (it2 != splitIoTensorInfo.adapterMap.end()) {
      auto &adapter = it2->second;
      if (adapter->tryOutCompleteUpstream()) {
        upstreamIo->outComplete(id);
        adapter->outLog("Called 'outComplete' on IStepIO");
      } else {
        // Can't complete the next adapter.
        adapter->outLog("Not yet able to call 'outComplete' on IStepIO");
        break;
      }
    } else {
      // We can't do much without a downstream adapter.
      throw error("[StepIOSplitter] No downstream adapter found for output "
                  "tensor {}@{}",
                  id,
                  splitIoTensorInfo.outIndex);
    }

    // Try the next index.
    outCompleteIndex = (outCompleteIndex + 1) % replicationFactor;
  }
}

} // namespace popart
