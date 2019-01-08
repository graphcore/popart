#ifndef GUARD_NEURALNET_IPUCOPY_HPP
#define GUARD_NEURALNET_IPUCOPY_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class IpuCopyOp : public Op {

  uint64_t destIpu;

public:
  IpuCopyOp(const OperatorIdentifier &_opid,
            Ir *_ir,
            uint64_t _destIpu,
            const std::string &name = "",
            const Attributes &_attr = {});
  void setup() final;

  uint64_t getDestIpu() { return destIpu; }
};
} // namespace poponnx

#endif
