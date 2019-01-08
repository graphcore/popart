#include <poponnx/op/ipucopy.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

IpuCopyOp::IpuCopyOp(const OperatorIdentifier &_opid,
                     Ir *_ir,
                     uint64_t _destIpu,
                     const std::string &name,
                     const Attributes &_attr)
    : Op(_opid, _ir, name, _attr), destIpu(_destIpu) {}

void IpuCopyOp::setup() { outInfo(0) = inInfo(0); }

// Have intentionally not added the IpuCopyOp to the OpManager. This IpuCopyOp
// needs to be explicitly created as part of the interipucopy transform

} // namespace poponnx
