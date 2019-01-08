#ifndef GUARD_NEURALNET_IPUCOPYX_HPP
#define GUARD_NEURALNET_IPUCOPYX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class IpuCopyOpx : public Opx {
public:
  IpuCopyOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
