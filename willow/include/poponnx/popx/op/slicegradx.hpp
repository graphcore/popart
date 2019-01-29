#ifndef GUARD_NEURALNET_SLICEGRADX_HPP
#define GUARD_NEURALNET_SLICEGRADX_HPP

#include <poponnx/popx/op/padx.hpp>

namespace poponnx {
namespace popx {

class SliceGradOpx : public PadOpx {
public:
  SliceGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
