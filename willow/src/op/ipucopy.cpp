#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
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

bool IpuCopyOp::isOutlineable() const {
  return getGraph().getIr().getSessionOptions().pingPongPhases > 1;
}

const SourceIpuMap &IpuCopyOp::getSourceIpus() const { return sourceIpus; }

const SourceTensorMap &IpuCopyOp::getSourceTensors() const {
  return sourceTensors;
}

uint64_t IpuCopyOp::getSourceIpu(const TensorId &tenId) const {
  return sourceIpus.at(tenId);
}

uint64_t IpuCopyOp::getSourceIpu() const {
  auto sourceIpu = sourceIpus.begin()->second;
  // check all source ipus are the same
  for (auto id_source : sourceIpus) {
    if (sourceIpu != id_source.second) {
      throw error("IpuCopyOp copies tensors from multiple sources: {}",
                  getFromToStr());
    }
  }
  return sourceIpu;
}

uint64_t IpuCopyOp::getMinSourceIpu() const {
  auto minSourceIpu = sourceIpus.begin()->second;

  for (auto id_source : sourceIpus) {
    if (minSourceIpu < id_source.second) {
      minSourceIpu = id_source.second;
    }
  }
  return minSourceIpu;
}

uint64_t IpuCopyOp::getMaxSourceIpu() const {
  auto maxSourceIpu = sourceIpus.begin()->second;

  for (auto id_source : sourceIpus) {
    if (maxSourceIpu > id_source.second) {
      maxSourceIpu = id_source.second;
    }
  }
  return maxSourceIpu;
}

void IpuCopyOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  // no appendAttribute for map<TensorId, uint64_t> so convert sourceIpus to
  // string
  std::set<int64_t> ipus;
  for (auto &sourceIpu : sourceIpus) {
    ipus.insert(sourceIpu.second);
  }

  os.appendAttribute("__sourceIpus", logging::format("{}", ipus));
  os.appendAttribute("__destIpu", destIpu);
}

bool IpuCopyOp::isIpuCopyOp() const { return true; }

bool IpuCopyOp::copiesOptimizerTensors() const {
  int optTensorCount = 0;
  for (auto tid_vgraphid : getSourceIpus()) {
    Tensor *t = getGraph().getTensors().get(tid_vgraphid.first);
    if (t->isOptimizerTensor()) {
      optTensorCount++;
    }
  }

  // Only return true if all tensors being copied
  // by op are optimizer tensors
  if (optTensorCount == getSourceIpus().size()) {
    return true;
  } else {
    return false;
  }
}

void IpuCopyOp::connectInTensor(InIndex inIndex,
                                TensorId tenId,
                                uint64_t sourceIpu) {
  sourceIpus.insert({tenId, sourceIpu});
  if (sourceTensors.find(sourceIpu) == sourceTensors.end()) {
    sourceTensors.insert({sourceIpu, {tenId}});
  } else {
    std::vector<TensorId> &tensorIds = sourceTensors.at(sourceIpu);
    tensorIds.push_back(tenId);
  }
  defaultConnectInTensor(inIndex, tenId);
}

void IpuCopyOp::disconnectInTensor(InIndex idx, Tensor *t) {
  auto sourceIpu = sourceIpus.at(t->id);
  sourceIpus.erase(t->id);

  auto &sourceIds = sourceTensors.at(sourceIpu);
  sourceIds.erase(std::remove(sourceIds.begin(), sourceIds.end(), t->id),
                  sourceIds.end());
  if (sourceIds.empty()) {
    sourceTensors.erase(sourceIpu);
  }

  Op::disconnectInTensor(idx, t);
}

// Have intentionally not added the IpuCopyOp to the OpManager. This IpuCopyOp
// needs to be explicitly created as part of the interipucopy transform

} // namespace popart
