#ifndef GUARD_NEURALNET_ALIASZEROCOPY_HPP
#define GUARD_NEURALNET_ALIASZEROCOPY_HPP

#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class AliasZeroCopy : public Transform {
public:
  static std::size_t id();

  AliasZeroCopy() : Transform() {}
  virtual ~AliasZeroCopy() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "AliasZeroCopy"; }

private:
  bool isSameTensor(Graph &graph, std::vector<Tensor *> tensors) const;
};

} // namespace popart

#endif
