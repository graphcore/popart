#ifndef GUARD_NEURALNET_IPUCOPY_HPP
#define GUARD_NEURALNET_IPUCOPY_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class IpuCopyOp : public Op {

  uint64_t sourceIpu;
  uint64_t destIpu;

public:
  IpuCopyOp(const OperatorIdentifier &_opid,
            uint64_t _sourceIpu,
            uint64_t _destIpu,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  uint64_t getDestIpu() const { return destIpu; }
  uint64_t getSourceIpu() const { return sourceIpu; }

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool isOutlineable() const override { return false; }
};
} // namespace poponnx

#endif
