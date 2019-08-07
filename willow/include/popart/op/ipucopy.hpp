#ifndef GUARD_NEURALNET_IPUCOPY_HPP
#define GUARD_NEURALNET_IPUCOPY_HPP

#include <popart/op.hpp>

namespace popart {

using SourceIpuMap    = std::map<TensorId, uint64_t>;
using SourceTensorMap = std::map<uint64_t, std::vector<TensorId>>;

class IpuCopyOp : public Op {
public:
  IpuCopyOp(const OperatorIdentifier &_opid,
            uint64_t _destIpu,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  uint64_t getDestIpu() const { return destIpu; }
  const SourceIpuMap &getSourceIpus() const;
  const SourceTensorMap &getSourceTensors() const;
  uint64_t getSourceIpu(const TensorId &tenId) const;
  uint64_t getSourceIpu() const;
  uint64_t getMinSourceIpu() const;
  uint64_t getMaxSourceIpu() const;

  void appendAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool isOutlineable() const override { return false; }

  bool isIpuCopyOp() const final;

  void connectInTensor(InIndex, TensorId, uint64_t sourceIpu);

  // A string of the form "[ sourceIpus ] --> [ destIpu ]"
  std::string getFromToStr() const;

private:
  void connectInTensor(InIndex, TensorId) override {
    throw error(
        "Must supply a sourceIpu when calling "
        "IpuCopyOp::connectInTensor(InIndex, TensorId, uint64_t sourceIpu)");
  }

  SourceIpuMap sourceIpus;
  SourceTensorMap sourceTensors;
  uint64_t destIpu;
};
} // namespace popart

#endif
