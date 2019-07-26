#ifndef GUARD_NEURALNET_SLICEGRADX_HPP
#define GUARD_NEURALNET_SLICEGRADX_HPP

#include <popart/popx/op/padx.hpp>

namespace popart {
namespace popx {

class SliceGradOpx : public PadOpx {
public:
  SliceGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
