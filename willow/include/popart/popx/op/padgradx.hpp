#ifndef GUARD_NEURALNET_PADGRADX_HPP
#define GUARD_NEURALNET_PADGRADX_HPP

#include <popart/popx/op/slicex.hpp>

namespace popart {

namespace popx {

class PadGradOpx : public SliceOpx {
public:
  PadGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
