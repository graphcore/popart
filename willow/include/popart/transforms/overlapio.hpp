// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OVERLAPIO_HPP
#define GUARD_NEURALNET_OVERLAPIO_HPP

#include <map>
#include <popart/op/exchange/exchange.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class OverlapIO : public Transform {
public:
  static std::size_t id();

  OverlapIO() : Transform() {}
  virtual ~OverlapIO() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "OverlapIO"; }

  /**
   * Check what level of \c ExchangeStrategy is required with overlapped IO.
   *
   * Each pipeline stage can contain IO operations that belong to any of the
   * strategies defined in the \c ExchangeStrategy enum. This will then inform
   * how the IO operations of each pipeline stages have to be unrolled.
   * \param ir IR to check for overlapped IO settings
   * \return   Map of required exchange strategies and pipeline stages in which
   *           exchanges occur. The set of stages will be empty if the
   *           \c ExchangeStrategy is not set on the
   *           \ref InputSettings or \ref AnchorReturnType of
   *           an input or output respectively.
   *           \c HostLoad and \c HostStore operations inserted by the
   *           \ref HostIoSetup transform will inherit the \c ExchangeStrategy
   *           from \ref InputSettings or \ref AnchorReturnType respectively.
   */
  static std::map<ExchangeStrategy, std::set<PipelineStage>>
  overlapIORequired(Ir &ir);
};

} // namespace popart

#endif
