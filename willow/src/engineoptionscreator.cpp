// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <engineoptionscreator.hpp>

namespace popart {

EngineOptionsCreator::EngineOptionsCreator(const SessionOptions &sessionOptions,
                                           const poplar::Target &target)
    : optionFlags{deriveOptionFlags(sessionOptions)},
      engineOptions{
          optionFlags,
          target,
          static_cast<unsigned>(sessionOptions.getGlobalReplicationFactor())} {}

const poplar::OptionFlags &EngineOptionsCreator::getOptionFlags() const {
  return optionFlags;
}

const poplar::EngineOptions &EngineOptionsCreator::getEngineOptions() const {
  return engineOptions;
}

poplar::OptionFlags
EngineOptionsCreator::deriveOptionFlags(const SessionOptions &sessionOptions) {
  poplar::OptionFlags engineOptions;

  if (sessionOptions.enablePrefetchDatastreams) {
    logging::devicex::info("Setting engine options for prefetch data streams "
                           "(exchange.streamBufferOverlap = hostRearrangeOnly, "
                           "exchange.enablePrefetch = true");
    engineOptions.set("exchange.streamBufferOverlap", "hostRearrangeOnly");
    engineOptions.set("exchange.enablePrefetch", "true");
  } else {
    engineOptions.set("exchange.enablePrefetch", "false");
  }

  if (sessionOptions.enableDistributedReplicatedGraphs) {
    logging::devicex::info("Setting firstRuntimeReplica {}",
                           sessionOptions.globalReplicaOffset);

    logging::devicex::info("Setting numberRuntimeReplica {}",
                           sessionOptions.replicatedGraphCount);

    std::string firstRuntimeReplica =
        std::to_string(sessionOptions.globalReplicaOffset);
    std::string numberRuntimeReplica =
        std::to_string(sessionOptions.replicatedGraphCount);

    engineOptions.set("target.syncReplicasIndependently", "true");
    engineOptions.set("target.firstRuntimeReplica", firstRuntimeReplica);
    engineOptions.set("target.numberRuntimeReplica", numberRuntimeReplica);
  }

  // The engine option `target.deterministicWorkers=true` ensures that random
  // behaviour is deterministic on all hardware but comes at the cost of
  // some performance. Note that we expect actual random Ops to be explicitly
  // seeded (so they are not affected) so the only time we actually need this
  // option is when the user enables stochastic rounding. We set this to
  // "false" when stochastic rounding is not enabled for a small performance
  // boost. Note that we avoid setting the option all together if the user
  // sets it explicitly.
  if (sessionOptions.engineOptions.find("target.deterministicWorkers") ==
      sessionOptions.engineOptions.end()) {
    auto detWorkerValue =
        (sessionOptions.enableStochasticRounding) ? "true" : "false";
    engineOptions.set("target.deterministicWorkers", detWorkerValue);
  }

  for (auto it : sessionOptions.engineOptions) {
    logging::devicex::info(
        "Setting engine option {} = {}", it.first, it.second);
    engineOptions.set(it.first, it.second);
  }

  return engineOptions;
}

} // namespace popart
