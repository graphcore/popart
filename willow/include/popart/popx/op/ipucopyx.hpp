#ifndef GUARD_NEURALNET_IPUCOPYX_HPP
#define GUARD_NEURALNET_IPUCOPYX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class IpuCopyOpx : public Opx {
public:
  IpuCopyOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // When pipelining is enabled, `IpuCopyOpx::grow` is not used.
  // `createPipelinedOutput` is used in place of grow, and created the
  // destination tensor for the copy.
  void createPipelinedOutput() const;
  // `growPipelined` add the copy program to the input Sequence. This is called
  // for every pipeline cycle the copy appears in.
  void growPipelined(poplar::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif
