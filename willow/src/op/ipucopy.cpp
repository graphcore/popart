#include <poponnx/op/ipucopy.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

IpuCopyOp::IpuCopyOp(const OperatorIdentifier &_opid,
                     uint64_t _sourceIpu,
                     uint64_t _destIpu,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), sourceIpu(_sourceIpu), destIpu(_destIpu) {}

void IpuCopyOp::setup() { outInfo(0) = inInfo(0); }

void IpuCopyOp::appendAttributes(std::stringstream &ss,
                                 const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "__sourceIpu", sourceIpu);
  appendAttribute(ss, tab, "__destIpu", destIpu);
}

// Have intentionally not added the IpuCopyOp to the OpManager. This IpuCopyOp
// needs to be explicitly created as part of the interipucopy transform

} // namespace poponnx
