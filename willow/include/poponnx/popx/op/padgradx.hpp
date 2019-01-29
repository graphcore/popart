#ifndef GUARD_NEURALNET_PADGRADX_HPP
#define GUARD_NEURALNET_PADGRADX_HPP

#include <poponnx/popx/op/slicex.hpp>

namespace poponnx {

namespace popx {

class PadGradOpx : public SliceOpx {
public:
  PadGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
