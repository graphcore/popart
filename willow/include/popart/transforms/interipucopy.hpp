// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INTERIPUCOPY_HPP
#define GUARD_NEURALNET_INTERIPUCOPY_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class InterIpuCopy : public Transform {
public:
  static std::size_t id();

  InterIpuCopy() : Transform() {}
  virtual ~InterIpuCopy() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "InterIpuCopy"; }

private:
  // Generate a name for the new tensor on the toIpu
  TensorId generateCopiedTensorId(Tensor *tensor, int64_t toIpu) const;

  // Used to add an insert IpuCopy op between ops that are on different IPUs
  void insertIpuCopy(Graph &graph,
                     Tensor *tensor,
                     Op *fromOp,
                     int64_t fromIpu,
                     Op *toOp,
                     int64_t toIpu) const;

  // Used to connect an tensor has already been copied between ipus by a
  // previous IpuCopy op
  void connectIpuCopy(Graph &graph,
                      Tensor *tensor,
                      Op *fromOp,
                      int64_t fromIpu,
                      Op *toOp,
                      int64_t toIpu) const;
};

} // namespace popart

#endif
