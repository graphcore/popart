#include <poponnx/op/ipucopy.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

IpuCopyOp::IpuCopyOp(const OperatorIdentifier &_opid,
                     uint64_t _sourceIpu,
                     uint64_t _destIpu,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), sourceIpu(_sourceIpu), destIpu(_destIpu) {
  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void IpuCopyOp::setup() {
  for (auto &idx_tensor : input->tensorMap()) {
    auto idx     = idx_tensor.first;
    outInfo(idx) = inInfo(idx);
  }
}

void IpuCopyOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("__sourceIpu", sourceIpu);
  os.appendAttribute("__destIpu", destIpu);
}

// Have intentionally not added the IpuCopyOp to the OpManager. This IpuCopyOp
// needs to be explicitly created as part of the interipucopy transform

} // namespace poponnx
