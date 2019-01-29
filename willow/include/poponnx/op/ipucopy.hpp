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
  void setup() final;

  uint64_t getDestIpu() { return destIpu; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;
};
} // namespace poponnx

#endif
