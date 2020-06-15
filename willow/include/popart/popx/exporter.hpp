// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPORTER_HPP
#define GUARD_NEURALNET_EXPORTER_HPP

#include <string>
#include <vector>

namespace poplar {
class Executable;
class OptionFlags;
} // namespace poplar

namespace popart {
class IStepIO;
class Builder;

namespace popx {
class Devicex;

// Return true if suport for exporters is compiled in.
bool exporterIsAvailable();

void exportWeights(const Devicex &device, const std::string &weightsPath);

void exportExecutable(poplar::Executable &executable,
                      const Devicex &device,
                      const poplar::OptionFlags &engineOptions,
                      const poplar::OptionFlags &deviceOptions,
                      const std::string &deviceHash,
                      int64_t numIPUs,
                      const std::string &executablePath);

void exportStepIO(IStepIO &step,
                  const Devicex &device,
                  int64_t numElements,
                  const std::string &outputFilename);
void exportStepIO(Builder &builder,
                  IStepIO &step,
                  int64_t numElements,
                  const std::vector<std::string> &feeds,
                  const std::string &outputFilename,
                  const std::string &metadataFilename);

} // namespace popx
} // namespace popart

#endif
