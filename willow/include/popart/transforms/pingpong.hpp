// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PINGPONG_HPP
#define GUARD_NEURALNET_PINGPONG_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class PingPong : public Transform {
public:
  static std::size_t id(int);

  PingPong(int pass_) : Transform(), pass(pass_) {}
  virtual ~PingPong() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(pass); }

  virtual std::string getName() const final {
    return "PingPong " + std::to_string(pass);
  }

private:
  TensorId generateInitTensorId(Tensor *tensor) const;

  TensorId generateLoadedTensorId(Tensor *tensor, int64_t load_index) const;

  TensorId generateGatheredTensorId(Tensor *tensor, int64_t load_index) const;

  TensorId generateRemoteArgTensorId(TensorId tid, VGraphId vgid) const;

  float costFn(Op *op) const;

  void verifyPlacementConsistency(const Op *op, unsigned num_stages) const;

  void verifyPingPongPhases(Graph &graph) const;

  void getModifiersInPhase(PingPongPhase phase,
                           Tensor *t,
                           std::vector<Op *> &modifyingConsumerOps) const;

  void
  getAliasedModifiersInPhase(Graph &graph,
                             PingPongPhase phase,
                             Tensor *t,
                             std::vector<Op *> &modifyingConsumerOps) const;

  void sanitizePlacementAnnotation(const Graph &graph,
                                   Op *op,
                                   PingPongPhase phase,
                                   unsigned num_stages) const;

private:
  class TensorStatus {
  public:
    OptionalPingPongPhase producerPingPongPhase;
    OptionalVGraphId loadStoreVGID;
    std::set<PingPongPhase> livePhases;
    std::map<PingPongPhase, std::vector<Op *>> modifiersInPhase;
    std::map<PingPongPhase, std::vector<Op *>> consumersInPhase;
    std::map<PingPongPhase, std::pair<bool, bool>> loadStoreInPhase;
  };

  bool isValidTensorLocation(const TensorLocation tensorLocation) const;

  bool tooSmallForOffChip(const TensorLocationSettings &tensorLocationSettings,
                          Tensor *tensor) const;

  const char *tensorLocationToStr(const TensorLocation tensorLocation) const;

  TensorLocation determineTensorLocation(Graph &graph, Tensor *tensor) const;
  TensorStatus determineTensorStatus(Graph &graph,
                                     Tensor *tensor,
                                     const std::vector<Op *> &consumerOps,
                                     unsigned num_stages) const;

  static void
  logTensorStatus(const Tensor *tensor, int num_phases, TensorStatus &status);

  static std::vector<Op *> getSortedConsumerOps(const Tensor *tensor);

  int pass;
};

} // namespace popart

#endif
