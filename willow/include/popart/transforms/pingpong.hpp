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

  TensorId generateCacheArgTensorId(TensorId tid, VGraphId vgid) const;

  float costFn(Op *op) const;

  void verifyPlacementConsistency(const Op *op,
                                  const unsigned num_stages) const;

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
  bool isValidCacheType(const CacheType cacheType) const;

  bool tooSmallForOffChip(const CacheSettings &cacheSettings,
                          Tensor *tensor) const;

  const char *cacheTypeToStr(const CacheType cacheType) const;

  CacheType determineCacheType(Graph &graph, Tensor *tensor) const;

  int pass;
};

} // namespace popart

#endif
