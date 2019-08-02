#include <memory>
#include <popart/op/ipucopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

std::string IpuCopyOp::getFromToStr() const {
  std::ostringstream ss;
  ss << "[ ";
  for (auto x : getSourceIpus()) {
    ss << x.second << " ";
  }
  ss << "] --> [ " << getDestIpu() << " ] ";
  return ss.str();
}

IpuCopyOp::IpuCopyOp(const OperatorIdentifier &_opid,
                     uint64_t _destIpu,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), destIpu(_destIpu) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

std::unique_ptr<Op> IpuCopyOp::clone() const {
  return std::make_unique<IpuCopyOp>(*this);
}

void IpuCopyOp::setup() {
  for (auto &idx_tensor : input->tensorMap()) {
    auto idx     = idx_tensor.first;
    outInfo(idx) = inInfo(idx);
  }
}

const SourceIpuMap &IpuCopyOp::getSourceIpus() const { return sourceIpus; }

uint64_t IpuCopyOp::getSourceIpu(const TensorId &tenId) const {
  return sourceIpus.at(tenId);
}

uint64_t IpuCopyOp::getSourceIpu() const {
  auto sourceIpu = sourceIpus.begin()->second;
  // check all source ipus are the same
  for (auto id_source : sourceIpus) {
    if (sourceIpu != id_source.second) {
      throw error("IpuCopyOp copies tensors from multiple sources");
    }
  }
  return sourceIpu;
}

void IpuCopyOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  // no appendAttribute for map<TensorId, uint64_t> so convert sourceIpus to
  // string
  os.appendAttribute("__sourceIpus", fmt::format("{}", sourceIpus));
  os.appendAttribute("__destIpu", destIpu);
}

bool IpuCopyOp::isIpuCopyOp() const { return true; }

void IpuCopyOp::connectInTensor(InIndex inIndex,
                                TensorId tenId,
                                uint64_t sourceIpu) {
  sourceIpus.insert({tenId, sourceIpu});
  defaultConnectInTensor(inIndex, tenId);
}

// Have intentionally not added the IpuCopyOp to the OpManager. This IpuCopyOp
// needs to be explicitly created as part of the interipucopy transform

} // namespace popart
